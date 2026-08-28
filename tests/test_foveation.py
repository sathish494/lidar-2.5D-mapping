"""
Unit tests for Dynamic Foveation Engine (Milestone 6).
Validates speed stretch, steering shear, and fine radius boundary deformation.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.foveation import (
    stretch_factor,
    shear_factor,
    fine_radius_at_angle,
    BASE_FINE_RADIUS,
    MAX_STRETCH,
    STRETCH_SPEED_REF,
    SHEAR_STRENGTH,
)
from src.grid.grid_types import VehicleState


def test_stationary_vehicle_fine_radius():
    """Stationary vehicle: fine_radius_at_angle() == BASE_FINE_RADIUS (10.0m) across all angles."""
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)

    for theta_deg in range(-180, 180, 15):
        theta_rad = np.deg2rad(theta_deg)
        radius = fine_radius_at_angle(theta_rad, state)
        assert radius == pytest.approx(BASE_FINE_RADIUS, abs=1e-4)


def test_speed_stretch_forward_vs_backward():
    """
    At STRETCH_SPEED_REF (20 m/s):
    - Forward (theta=0): fine radius == BASE_FINE_RADIUS * MAX_STRETCH (25.0m).
    - Half speed (10 m/s): fine radius == BASE_FINE_RADIUS * 1.75 (17.5m).
    - Behind vehicle (theta = pi): fine radius == BASE_FINE_RADIUS (10.0m, no forward stretch).
    """
    state_top_speed = VehicleState(speed_mps=STRETCH_SPEED_REF, steering_angle_rad=0.0)
    
    # Forward dead ahead (theta = 0)
    r_forward = fine_radius_at_angle(0.0, state_top_speed)
    assert r_forward == pytest.approx(BASE_FINE_RADIUS * MAX_STRETCH, abs=1e-4)

    # Half speed (10 m/s)
    state_mid_speed = VehicleState(speed_mps=10.0, steering_angle_rad=0.0)
    r_mid = fine_radius_at_angle(0.0, state_mid_speed)
    assert r_mid == pytest.approx(BASE_FINE_RADIUS * 1.75, abs=1e-4)

    # Behind vehicle (theta = pi or -pi)
    r_back = fine_radius_at_angle(np.pi, state_top_speed)
    assert r_back == pytest.approx(BASE_FINE_RADIUS, abs=1e-4)

    # Overspeed cap (30 m/s > 20 m/s)
    state_overspeed = VehicleState(speed_mps=30.0, steering_angle_rad=0.0)
    r_over = fine_radius_at_angle(0.0, state_overspeed)
    assert r_over == pytest.approx(BASE_FINE_RADIUS * MAX_STRETCH, abs=1e-4)


def test_steering_lateral_shear():
    """
    Steering right (steering_angle > 0):
    - Right turn quadrant has expanded fine radius.
    - Left quadrant has reduced/smaller fine radius.
    """
    steer_right = np.deg2rad(30.0)  # +30 deg right
    state = VehicleState(speed_mps=0.0, steering_angle_rad=steer_right)

    # Check lateral sides (+pi/2 right vs -pi/2 left)
    r_right_side = fine_radius_at_angle(np.pi / 2.0, state)
    r_left_side = fine_radius_at_angle(-np.pi / 2.0, state)

    assert r_right_side > BASE_FINE_RADIUS
    assert r_left_side < BASE_FINE_RADIUS
    assert r_right_side > r_left_side

    # Inside turn angle (+30 deg) has higher radius than same negative angle (-30 deg)
    r_turn_pos = fine_radius_at_angle(steer_right, state)
    r_turn_neg = fine_radius_at_angle(-steer_right, state)
    assert r_turn_pos > r_turn_neg
