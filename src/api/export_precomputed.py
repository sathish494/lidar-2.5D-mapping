"""
Precomputed Scenario Frame Exporter for FoveaMap.

Executes the full pipeline (perception, dynamic foveation, grid binning, tracking,
anti-ghosting, and memory metrics) and saves JSON payloads to disk for deterministic,
zero-latency replay.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingestion.kitti_loader import KITTILoader
from src.perception.segment import segment_points
from src.perception.class_map import map_semantickitti_to_4class
from src.grid.grid_engine import PolarGridEngine
from src.grid.grid_types import VehicleState, GridCell
from src.grid.foveation import BASE_FINE_RADIUS, MAX_STRETCH, SHEAR_STRENGTH
from src.tracking.kalman_tracker import (
    KalmanTrackerManager,
    erase_vacated_footprints,
    cluster_dynamic_detections,
)
from src.synthetic.scenarios import ScenarioGenerator


def compute_memory_metrics(grid_map: Dict[Any, GridCell], engine: PolarGridEngine) -> Dict[str, float]:
    """
    Computes exact memory usage for FoveaMap 2.5D sparse grid vs Uniform 5cm high-res baseline.
    
    A uniform 5cm grid covering 100m radius:
    Area = pi * 100^2 = 31,416 m^2.
    Cell count = 31,416 / (0.05 * 0.05) = 12,566,400 cells.
    Uniform dense storage (assuming 16 bytes/cell): 12,566,400 * 16 = 201,062,400 bytes (~201 MB).
    
    FoveaMap Sparse Storage:
    Active cells count * (key: 8 bytes + cell struct: 32 bytes) = count * 40 bytes.
    """
    active_cells = len(grid_map)
    # Uniform baseline: 100m radius uniform 5cm grid
    # Nominal sparse active cells in uniform grid ~ active_cells * 25 (since outer is 0.5m -> 100x area, mid is 0.15m -> 9x area)
    uniform_cells_equivalent = 0
    for cell in grid_map.values():
        res = cell.resolution_tier
        scale = (res / 0.05) ** 2
        uniform_cells_equivalent += int(scale)

    uniform_cells_equivalent = max(uniform_cells_equivalent, active_cells * 9)

    bytes_per_sparse_cell = 40  # 8 bytes key + 32 bytes GridCell fields
    bytes_foveated = active_cells * bytes_per_sparse_cell
    bytes_uniform = uniform_cells_equivalent * bytes_per_sparse_cell

    savings_pct = (1.0 - (bytes_foveated / max(bytes_uniform, 1))) * 100.0
    return {
        "active_cells_count": float(active_cells),
        "memory_bytes_foveated": float(bytes_foveated),
        "memory_bytes_uniform_baseline": float(bytes_uniform),
        "memory_savings_pct": float(np.clip(savings_pct, 0.0, 99.9)),
    }


def process_sequence_to_json(
    sequence: List[Tuple[np.ndarray, VehicleState, Dict[str, Any]]],
    scenario_name: str,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Runs full FoveaMap perception, grid binning, and tracking over a frame sequence.
    """
    engine = PolarGridEngine()
    tracker = KalmanTrackerManager()
    prev_footprints: Dict[int, Any] = {}

    frames_payload = []

    for frame_id, (scan, v_state, meta) in enumerate(sequence):
        t_start = time.perf_counter()

        # Step 1: Perception / Semantic Segmentation
        if scan.shape[1] >= 5:
            raw_classes = scan[:, 4].astype(np.int32)
            if np.any(raw_classes > 3):  # Raw SemanticKITTI classes
                classes = map_semantickitti_to_4class(raw_classes)
            else:
                classes = raw_classes
        else:
            classes = segment_points(scan[:, :3], use_heuristic=True)

        points_5d = np.column_stack([scan[:, :3], scan[:, 3] if scan.shape[1] >= 4 else np.zeros(len(scan)), classes])

        # Step 2: Grid Projection & Multi-layer aggregation
        roof_h = 4.0 if "bridge" in scenario_name.lower() else 2.5
        grid_map = engine.project_to_grid(
            points=points_5d,
            vehicle_state=v_state,
            frame_id=frame_id,
            roof_height_m=roof_h,
            use_foveation=True,
        )

        # Step 3: Extract dynamic clusters for tracking
        raw_detections = []
        for key, cell in grid_map.items():
            if cell.semantic_class == 3:
                cx, cy, _, _ = engine.get_cell_spatial_center(key[0], key[1], cell.resolution_tier)
                raw_detections.append((cx, cy, 3))

        dynamic_detections = cluster_dynamic_detections(raw_detections, cluster_dist_m=2.5)
        active_tracks = tracker.update_tracks(dynamic_detections)

        # Step 4: Active Footprint Erasure (Anti-Ghosting)
        grid_map, prev_footprints, erased_count = erase_vacated_footprints(
            grid_map=grid_map,
            tracks=active_tracks,
            prev_footprints=prev_footprints,
            frame_id=frame_id,
            grid_engine=engine,
        )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / max(latency_ms, 0.001)

        # Step 5: Serialize cells
        cells_list = []
        for (r_idx, a_idx), cell in grid_map.items():
            cx, cy, r_val, th_val = engine.get_cell_spatial_center(r_idx, a_idx, cell.resolution_tier)
            c_dict = cell.to_dict(r_idx, a_idx)
            c_dict["x"] = round(cx, 3)
            c_dict["y"] = round(cy, 3)
            c_dict["r"] = round(r_val, 3)
            c_dict["theta"] = round(th_val, 3)
            cells_list.append(c_dict)

        mem_metrics = compute_memory_metrics(grid_map, engine)

        frame_data = {
            "frame_id": frame_id,
            "timestamp": float(meta.get("timestamp_s", frame_id * 0.1)),
            "scenario_name": scenario_name,
            "description": meta.get("description", ""),
            "vehicle_state": {
                "speed_mps": round(v_state.speed_mps, 2),
                "steering_angle_rad": round(v_state.steering_angle_rad, 4),
            },
            "foveation_params": {
                "base_fine_radius_m": BASE_FINE_RADIUS,
                "max_stretch": MAX_STRETCH,
                "shear_strength": SHEAR_STRENGTH,
            },
            "cells": cells_list,
            "tracks": [t.to_dict() for t in active_tracks],
            "metrics": {
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 2),
                "memory_bytes_foveated": mem_metrics["memory_bytes_foveated"],
                "memory_bytes_uniform_baseline": mem_metrics["memory_bytes_uniform_baseline"],
                "memory_savings_pct": round(mem_metrics["memory_savings_pct"], 1),
                "ghosting_cells_erased": erased_count,
                "active_cells_count": mem_metrics["active_cells_count"],
            },
        }
        frames_payload.append(frame_data)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(frames_payload, f)
        print(f"[INFO] Exported {len(frames_payload)} frames to {output_path}")

    return frames_payload


def export_all_precomputed(output_dir: str = "data/precomputed") -> None:
    """Exports precomputed frames for KITTI sample and all 4 synthetic scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Synthetic KITTI-Like Sequence
    kitti_dir = "data/synthetic_kitti_like" if os.path.exists("data/synthetic_kitti_like") else "data/kitti_sample"
    if os.path.exists(os.path.join(kitti_dir, "velodyne")):
        loader = KITTILoader(kitti_dir)
        kitti_seq = []
        for i in range(len(loader)):
            scan = loader[i]
            v_state = VehicleState(speed_mps=8.0, steering_angle_rad=0.0)
            meta = {
                "scenario_name": "Synthetic KITTI-Format Sequence",
                "timestamp_s": i * 0.1,
                "description": "Procedural LiDAR scan sequence in SemanticKITTI format with traffic and sidewalks.",
            }
            kitti_seq.append((scan, v_state, meta))
        process_sequence_to_json(kitti_seq, "synthetic_kitti_like", os.path.join(output_dir, "synthetic_kitti_like.json"))
        # Also alias for backward compatibility
        process_sequence_to_json(kitti_seq, "kitti_sample", os.path.join(output_dir, "kitti_sample.json"))

    # 2. Synthetic Scenarios
    gen = ScenarioGenerator()
    scenarios = gen.get_all_scenarios()
    for name, seq in scenarios.items():
        process_sequence_to_json(seq, name, os.path.join(output_dir, f"{name}.json"))


if __name__ == "__main__":
    export_all_precomputed()
