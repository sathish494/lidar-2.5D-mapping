"""
Unit tests for static resolution tiers (Milestone 2).
Validates get_resolution(), get_tier_index(), and angular_step().
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.resolution import get_resolution, get_tier_index, angular_step, RESOLUTION_TIERS


def test_resolution_tiers_static():
    """Validates get_resolution() at ranges 5m, 15m, 60m according to Section 4."""
    # Radius < 10m -> 5cm (0.05m)
    assert get_resolution(0.0) == pytest.approx(0.05)
    assert get_resolution(5.0) == pytest.approx(0.05)
    assert get_resolution(9.99) == pytest.approx(0.05)
    assert get_tier_index(5.0) == 0

    # 10m <= Radius < 30m -> 15cm (0.15m)
    assert get_resolution(10.0) == pytest.approx(0.15)
    assert get_resolution(15.0) == pytest.approx(0.15)
    assert get_resolution(29.99) == pytest.approx(0.15)
    assert get_tier_index(15.0) == 1

    # Radius >= 30m -> 50cm (0.50m)
    assert get_resolution(30.0) == pytest.approx(0.50)
    assert get_resolution(60.0) == pytest.approx(0.50)
    assert get_resolution(99.0) == pytest.approx(0.50)
    assert get_resolution(150.0) == pytest.approx(0.50)
    assert get_tier_index(60.0) == 2


def test_angular_step():
    """Validates that arc length approx equals target resolution."""
    # At range 5m, target res 0.05m -> angular step = 0.05 / 5.0 = 0.01 rad
    step_5m = angular_step(0.05, 5.0)
    assert step_5m == pytest.approx(0.01)
    # Arc length at 5m = 5.0 * 0.01 = 0.05m
    assert step_5m * 5.0 == pytest.approx(0.05)

    # Near-origin clamping at range 0.1m with min_range 0.5m
    step_origin = angular_step(0.05, 0.1, min_range_m=0.5)
    assert step_origin == pytest.approx(0.05 / 0.5)
