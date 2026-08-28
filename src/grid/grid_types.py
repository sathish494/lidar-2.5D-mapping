"""
Core Data Structures for FoveaMap Grid Engine.

Defines PointCloud type, GridCell dataclass, GridMap dict type, and VehicleState.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
import numpy as np

# PointCloud array format: (N, 5) float32: [x, y, z, intensity, class_id]
PointCloud = np.ndarray


@dataclass
class GridCell:
    """
    Represents a 2.5D column in the polar grid.
    """
    elevation_ground: float                     # z_min of surface (drivable or non-drivable)
    elevation_obstacle_bottom: Optional[float]  # None if no overhang detected
    elevation_obstacle_top: Optional[float]     # None if no overhang detected
    semantic_class: int                         # 0=Drivable, 1=Non-Drivable, 2=Static, 3=Dynamic, -1=Unknown
    point_count: int                            # Number of LiDAR points contributing to this cell
    last_updated_frame: int                     # Timestamp / frame counter for temporal decay & hysteresis
    resolution_tier: float                      # Cell size in meters (e.g. 0.05, 0.15, 0.50)
    confidence: float = 1.0                     # Confidence score (1.0 = high, 0.3 = degraded/sparse)

    def to_dict(self, ring_idx: int, angle_idx: int) -> Dict[str, Any]:
        """Convert cell to JSON-serializable dictionary matching API schema."""
        return {
            "ring_idx": ring_idx,
            "angle_idx": angle_idx,
            "elevation_ground": float(self.elevation_ground),
            "elevation_obstacle_bottom": float(self.elevation_obstacle_bottom) if self.elevation_obstacle_bottom is not None else None,
            "elevation_obstacle_top": float(self.elevation_obstacle_top) if self.elevation_obstacle_top is not None else None,
            "semantic_class": int(self.semantic_class),
            "confidence": float(self.confidence),
            "resolution_tier": float(self.resolution_tier),
            "point_count": int(self.point_count),
        }


# Sparse dictionary mapping (ring_idx, angle_idx) -> GridCell
GridMap = Dict[Tuple[int, int], GridCell]


@dataclass
class VehicleState:
    """
    State of ego vehicle for dynamic foveation calculations.
    """
    speed_mps: float = 0.0
    steering_angle_rad: float = 0.0  # Positive = right turn, negative = left turn
