"""
Unit tests for PolarGridEngine baseline projection and Z-clipping (Milestones 3 and 5).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.grid_engine import PolarGridEngine
from src.grid.grid_types import VehicleState


def test_grid_engine_baseline_projection():
    """Validates that synthetic points land in expected cells and correct elevation_ground."""
    engine = PolarGridEngine()
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)

    # 5 synthetic points inside the exact same 5cm cell [5.00, 5.05) near x=5.02m, y=0.0m
    # Ground surface around z=-0.5m with min z = -0.52m and class 0 (Drivable)
    points = np.array([
        [5.02, 0.00, -0.50, 0.5, 0],
        [5.03, 0.00, -0.48, 0.6, 0],
        [5.01, 0.00, -0.52, 0.4, 0],  # Min z in this cell
        [5.02, 0.00, -0.49, 0.5, 0],
        [5.04, 0.00, -0.47, 0.5, 1],  # 1 non-drivable point (minority)
    ], dtype=np.float32)

    grid_map = engine.project_to_grid(points, vehicle_state=state, frame_id=1, use_foveation=False)

    assert len(grid_map) >= 1
    # Check that at least one cell contains our cluster
    found_cell = False
    for key, cell in grid_map.items():
        if cell.point_count >= 3:
            found_cell = True
            assert cell.elevation_ground == pytest.approx(-0.52, abs=1e-3)
            assert cell.elevation_obstacle_bottom is None
            assert cell.elevation_obstacle_top is None
            assert cell.semantic_class == 0  # Majority vote: 4 class 0 vs 1 class 1
            assert cell.resolution_tier == pytest.approx(0.05)
            assert cell.last_updated_frame == 1
    assert found_cell, "Expected aggregated cell not found in GridMap"


def test_z_clipping_roof_height():
    """Validates Milestone 5: points above roof_height_m are excluded entirely."""
    engine = PolarGridEngine()
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)

    # 3 points: z=-1.0 (valid), z=2.0 (valid <= 2.5), z=3.0 (invalid > 2.5 roof height)
    points = np.array([
        [4.0, 0.0, -1.0, 0.5, 0],
        [4.0, 0.0, 2.0, 0.5, 2],
        [4.0, 0.0, 3.0, 0.5, 2],  # Above roof height (2.5m default)
    ], dtype=np.float32)

    grid_map = engine.project_to_grid(
        points, vehicle_state=state, frame_id=1, roof_height_m=2.5, use_foveation=False
    )

    # The point at z=3.0 must be dropped. The remaining points have z=-1.0 and z=2.0
    for key, cell in grid_map.items():
        assert cell.point_count == 2
        # Max z in this cell cannot exceed 2.5
        if cell.elevation_obstacle_top is not None:
            assert cell.elevation_obstacle_top <= 2.5
        assert cell.elevation_ground == pytest.approx(-1.0)


def test_empty_and_out_of_range_points():
    """Validates handling of empty inputs and points outside [0.5m, 100m]."""
    engine = PolarGridEngine()
    empty_pts = np.empty((0, 5), dtype=np.float32)
    assert engine.project_to_grid(empty_pts) == {}

    # Point too close (r=0.2m) and point too far (r=150m)
    out_of_range_pts = np.array([
        [0.1, 0.1, 0.0, 0.5, 0],    # r ~ 0.14m < 0.5m
        [120.0, 0.0, 0.0, 0.5, 0],  # r = 120m > 100m
    ], dtype=np.float32)
    assert engine.project_to_grid(out_of_range_pts) == {}
