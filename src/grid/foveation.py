"""
Dynamic Foveation Module for FoveaMap.

Calculates dynamic deformation of the high-resolution (Tier 0) fine zone
based on vehicle speed (forward elongation) and steering angle (lateral shear).
"""

import os
from typing import Optional, Dict, Any, Union
import numpy as np
import yaml

from src.grid.grid_types import VehicleState


def load_foveation_config(config_path: str = "configs/default.yaml") -> Dict[str, float]:
    """Loads foveation constants from default.yaml."""
    defaults = {
        "base_fine_radius_m": 10.0,
        "max_stretch": 2.5,
        "stretch_speed_ref_mps": 20.0,
        "shear_strength": 0.6,
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                f_cfg = cfg.get("foveation", {})
                for k in defaults:
                    if k in f_cfg:
                        defaults[k] = float(f_cfg[k])
        except Exception:
            pass
    return defaults


_FOVEA_CFG = load_foveation_config()

BASE_FINE_RADIUS: float = _FOVEA_CFG["base_fine_radius_m"]
MAX_STRETCH: float = _FOVEA_CFG["max_stretch"]
STRETCH_SPEED_REF: float = _FOVEA_CFG["stretch_speed_ref_mps"]
SHEAR_STRENGTH: float = _FOVEA_CFG["shear_strength"]


def stretch_factor(
    speed_mps: float,
    max_stretch: float = MAX_STRETCH,
    speed_ref_mps: float = STRETCH_SPEED_REF,
) -> float:
    """Linear ramp 1.0 -> max_stretch at speed_ref_mps, capped."""
    clamped_speed = max(0.0, float(speed_mps))
    ratio = min(clamped_speed / speed_ref_mps, 1.0)
    return float(1.0 + (max_stretch - 1.0) * ratio)


def shear_factor(
    theta_rad: Union[float, np.ndarray],
    steering_angle_rad: float,
    shear_strength: float = SHEAR_STRENGTH,
) -> Union[float, np.ndarray]:
    """Lateral shear factor supporting both float and numpy arrays."""
    steer_norm = float(np.clip(steering_angle_rad / np.pi, -1.0, 1.0))
    alignment = np.cos(theta_rad - steering_angle_rad)
    return 1.0 + shear_strength * steer_norm * alignment


def fine_radius_at_angle(
    theta_rad: Union[float, np.ndarray],
    state: VehicleState,
    base_radius: float = BASE_FINE_RADIUS,
    max_stretch: float = MAX_STRETCH,
    speed_ref_mps: float = STRETCH_SPEED_REF,
    shear_strength: float = SHEAR_STRENGTH,
) -> Union[float, np.ndarray]:
    """Computes deformed fine radius for scalar or vectorized theta array."""
    is_array = isinstance(theta_rad, np.ndarray)
    speed_factor = stretch_factor(state.speed_mps, max_stretch, speed_ref_mps)

    if is_array:
        forward_bias = np.where(np.abs(theta_rad) <= (np.pi / 2.0), speed_factor, 1.0)
        steer_norm = float(np.clip(state.steering_angle_rad / np.pi, -1.0, 1.0))
        alignment = np.cos(theta_rad - state.steering_angle_rad)
        lateral_bias = 1.0 + shear_strength * steer_norm * alignment
        radius = base_radius * forward_bias * lateral_bias
        return np.maximum(radius, 1.0)
    else:
        forward_bias = speed_factor if abs(theta_rad) <= (np.pi / 2.0) else 1.0
        steer_norm = float(np.clip(state.steering_angle_rad / np.pi, -1.0, 1.0))
        alignment = float(np.cos(theta_rad - state.steering_angle_rad))
        lateral_bias = 1.0 + shear_strength * steer_norm * alignment
        radius = base_radius * forward_bias * lateral_bias
        return float(max(radius, 1.0))
