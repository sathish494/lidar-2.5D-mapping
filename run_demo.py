"""
FoveaMap One-Command Demo Launcher.

Usage:
  python run_demo.py             # Launches server and opens browser dashboard
  python run_demo.py --eval      # Runs evaluation and benchmarks report
  python run_demo.py --export    # Recomputes and exports all scenario JSONs
"""

import os
import sys
import argparse
import webbrowser
import uvicorn
import yaml

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.sample_generator import generate_kitti_sample_dataset
from src.api.export_precomputed import export_all_precomputed
from src.metrics.evaluate import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="FoveaMap 3D->2.5D Perception Pipeline")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation and print report")
    parser.add_argument("--export", action="store_true", help="Re-export precomputed scenario JSONs")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    # Step 1: Ensure sample data exists
    kitti_dir = "data/synthetic_kitti_like"
    if not os.path.exists(os.path.join(kitti_dir, "velodyne")):
        print("[INFO] Generating synthetic KITTI-format procedural sequence in data/synthetic_kitti_like...")
        generate_kitti_sample_dataset(kitti_dir, num_frames=30)

    # Step 2: Handle --eval flag
    if args.eval:
        run_full_evaluation()
        return

    # Step 3: Handle --export flag
    if args.export:
        print("[INFO] Exporting all precomputed scenario frames...")
        export_all_precomputed()
        return

    # Ensure precomputed files exist
    precomputed_kitti = "data/precomputed/synthetic_kitti_like.json"
    if not os.path.exists(precomputed_kitti):
        print("[INFO] Generating initial precomputed scenario frames...")
        export_all_precomputed()

    # Step 4: Launch Web Server & Dashboard
    dashboard_url = f"http://localhost:{args.port}/"
    print("\n" + "=" * 70)
    print("      FOVEAMAP PERCEPTION DASHBOARD & PIPELINE RUNNING")
    print(f"      Dashboard URL : {dashboard_url}")
    print(f"      WebSocket API : ws://localhost:{args.port}/ws/grid")
    print(f"      REST Health   : http://localhost:{args.port}/health")
    print("=" * 70 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(dashboard_url)
        except Exception:
            pass

    from src.api.server import app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
