"""
FastAPI Server & WebSocket Streaming Service for FoveaMap.

Serves REST endpoints, WebSocket streaming, and static Three.js dashboard on port 8080.
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
import yaml
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from src.ingestion.kitti_loader import KITTILoader
from src.perception.segment import segment_points
from src.grid.grid_engine import PolarGridEngine
from src.grid.grid_types import VehicleState
from src.grid.foveation import fine_radius_at_angle, BASE_FINE_RADIUS, MAX_STRETCH, SHEAR_STRENGTH
from src.tracking.kalman_tracker import KalmanTrackerManager, erase_vacated_footprints
from src.synthetic.scenarios import ScenarioGenerator
from src.api.export_precomputed import process_sequence_to_json, compute_memory_metrics, export_all_precomputed

app = FastAPI(title="FoveaMap Perception API", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = "configs/default.yaml"
PRECOMPUTED_DIR = "data/precomputed"

_scenario_gen = ScenarioGenerator()
_grid_engine = PolarGridEngine()


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


@app.get("/health")
async def health_check():
    """Health check endpoint required by Section 7.3."""
    return {"status": "ok", "timestamp": time.time(), "service": "foveamap-api"}


@app.get("/api/config")
async def get_configuration():
    """Returns central pipeline configuration and class color mappings."""
    return JSONResponse(content=load_config())


@app.get("/api/scenarios")
async def list_scenarios():
    """Returns list of available demo scenarios."""
    scenarios = [
        {
            "id": "synthetic_kitti_like",
            "name": "Synthetic KITTI-Format Sequence (Procedural)",
            "description": "30 procedural sequential LiDAR frames in SemanticKITTI format with traffic, sidewalks, and buildings.",
            "type": "synthetic",
        },
        {
            "id": "urban_intersection",
            "name": "Urban Intersection",
            "description": "Dense cross-traffic, crossing pedestrian, roadside curbs, and turning ego vehicle.",
            "type": "synthetic",
        },
        {
            "id": "highway_cruise",
            "name": "Highway Cruise (22 m/s)",
            "description": "High-speed cruising with active forward foveation elongation up to 25m.",
            "type": "synthetic",
        },
        {
            "id": "pothole_alley",
            "name": "Pothole & Rough Terrain Alley",
            "description": "Distinguishes smooth drivable road (Class 0) from hazardous non-drivable potholes (Class 1).",
            "type": "synthetic",
        },
        {
            "id": "bridge_overpass",
            "name": "Bridge Overpass",
            "description": "Multi-layer underpass: ground surface (z=-1.5m) and bridge deck ceiling (z=1.8 to 3.2m).",
            "type": "synthetic",
        },
    ]
    return JSONResponse(content=scenarios)


@app.get("/api/frames/{scenario_id}")
async def get_scenario_frames(scenario_id: str):
    """Retrieves precomputed frame sequence for the requested scenario."""
    # Support backward-compatible id alias
    if scenario_id == "kitti_sample":
        scenario_id = "synthetic_kitti_like"

    cache_file = os.path.join(PRECOMPUTED_DIR, f"{scenario_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)

    # If cache not found, generate on the fly
    if scenario_id == "synthetic_kitti_like":
        loader = KITTILoader("data/synthetic_kitti_like")
        kitti_seq = []
        for i in range(len(loader)):
            scan = loader[i]
            v_state = VehicleState(speed_mps=8.0, steering_angle_rad=0.0)
            meta = {"scenario_name": "Synthetic KITTI-Format Sequence", "timestamp_s": i * 0.1}
            kitti_seq.append((scan, v_state, meta))
        frames = process_sequence_to_json(kitti_seq, "synthetic_kitti_like", cache_file)
        return JSONResponse(content=frames)
    else:
        scenarios = _scenario_gen.get_all_scenarios()
        if scenario_id in scenarios:
            frames = process_sequence_to_json(scenarios[scenario_id], scenario_id, cache_file)
            return JSONResponse(content=frames)

    raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")


@app.websocket("/ws/grid")
async def websocket_grid_stream(websocket: WebSocket):
    """
    Real-time WebSocket streaming endpoint conforming to Section 7.3.
    Streams 2.5D grid frames and handles live client parameter adjustments.
    """
    await websocket.accept()
    current_scenario = "synthetic_kitti_like"
    current_frame_idx = 0
    is_playing = True
    speed_mps = 8.0
    steering_rad = 0.0
    playback_fps = 10.0
    user_override = False

    # Load initial frames
    cache_file = os.path.join(PRECOMPUTED_DIR, f"{current_scenario}.json")
    if not os.path.exists(cache_file):
        export_all_precomputed()

    try:
        with open(cache_file, "r") as f:
            frames = json.load(f)
    except Exception:
        frames = []

    try:
        while True:
            # Check for incoming client messages (non-blocking)
            try:
                msg_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                msg = json.loads(msg_text)
                cmd = msg.get("command")

                if cmd == "set_scenario":
                    scenario_id = msg.get("scenario_id", "kitti_sample")
                    if scenario_id != current_scenario:
                        current_scenario = scenario_id
                        current_frame_idx = 0
                        user_override = False
                        f_path = os.path.join(PRECOMPUTED_DIR, f"{current_scenario}.json")
                        if os.path.exists(f_path):
                            with open(f_path, "r") as f:
                                frames = json.load(f)

                elif cmd == "set_params":
                    speed_mps = float(msg.get("speed_mps", speed_mps))
                    steering_rad = float(msg.get("steering_angle_rad", steering_rad))
                    user_override = True

                elif cmd == "play":
                    is_playing = True
                elif cmd == "pause":
                    is_playing = False
                elif cmd == "step":
                    is_playing = False
                    current_frame_idx = (current_frame_idx + 1) % max(len(frames), 1)
                elif cmd == "seek":
                    current_frame_idx = int(msg.get("frame_id", 0)) % max(len(frames), 1)

            except asyncio.TimeoutError:
                pass

            if frames and len(frames) > 0:
                frame_data = dict(frames[current_frame_idx])
                # Allow live speed/steering override
                if user_override:
                    frame_data["vehicle_state"]["speed_mps"] = speed_mps
                    frame_data["vehicle_state"]["steering_angle_rad"] = steering_rad

                await websocket.send_json(frame_data)

                if is_playing:
                    current_frame_idx = (current_frame_idx + 1) % len(frames)

            await asyncio.sleep(1.0 / playback_fps)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Disconnected: {e}")


# Mount dashboard directory for static frontend hosting
DASHBOARD_DIR = os.path.abspath("dashboard")
if os.path.exists(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", 8080))
    print(f"[INFO] Starting FoveaMap Unified Server on http://{host}:{port}", flush=True)
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)
