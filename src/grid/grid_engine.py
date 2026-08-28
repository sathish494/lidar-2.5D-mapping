"""
FoveaMap Polar Grid Engine.

Projects 3D LiDAR point clouds into a multi-resolution 2.5D polar grid with:
- Z-clipping (roof height filter)
- Dynamic foveation tier selection
- Multi-layer elevation aggregation (overhang detection)
- Semantic class majority voting
- Cell confidence and temporal tracking
"""

import os
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import yaml

from src.grid.grid_types import PointCloud, GridCell, GridMap, VehicleState
from src.grid.resolution import get_resolution, angular_step, RESOLUTION_TIERS
from src.grid.foveation import fine_radius_at_angle, BASE_FINE_RADIUS


class PolarGridEngine:
    """
    Core polar grid engine managing binning, elevation extraction, and 2.5D mapping.
    """
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        
        grid_cfg = self.config.get("grid", {})
        self.max_range_m: float = float(grid_cfg.get("max_range_m", 100.0))
        self.min_range_m: float = float(grid_cfg.get("min_range_m", 0.5))
        self.roof_height_m: float = float(grid_cfg.get("roof_height_m", 2.5))
        self.overhang_gap_threshold_m: float = float(grid_cfg.get("overhang_gap_threshold_m", 0.5))
        self.hysteresis_margin_m: float = float(grid_cfg.get("hysteresis_margin_m", 0.5))
        self.min_points_per_ring_sector: int = int(grid_cfg.get("min_points_per_ring_sector", 3))

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def compute_ring_angle_indices(
        self,
        r: np.ndarray,
        theta: np.ndarray,
        vehicle_state: Optional[VehicleState] = None,
        use_foveation: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (ring_idx, angle_idx, res_tier) for an array of range and angle points.
        
        Args:
            r: (N,) float array of ranges in meters.
            theta: (N,) float array of azimuth angles in radians [-pi, pi].
            vehicle_state: Current vehicle dynamic state.
            use_foveation: Whether to apply dynamic foveation fine radius deform.
            
        Returns:
            ring_indices: (N,) int32 array.
            angle_indices: (N,) int32 array.
            res_tiers: (N,) float32 array of cell resolution (0.05, 0.15, 0.50).
        """
        n_points = len(r)
        if n_points == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )

        res_tiers = np.zeros(n_points, dtype=np.float32)

        if use_foveation and vehicle_state is not None:
            # Dynamic fine radius per point angle (vectorized)
            fine_radii = fine_radius_at_angle(theta, vehicle_state)
        else:
            fine_radii = np.full(n_points, BASE_FINE_RADIUS, dtype=np.float32)

        # Tier 0 (< fine_radii): 0.05m
        # Tier 1 (fine_radii <= r < 30.0m): 0.15m
        # Tier 2 (r >= 30.0m): 0.50m
        tier0_mask = r < fine_radii
        tier1_mask = (~tier0_mask) & (r < 30.0)
        tier2_mask = (~tier0_mask) & (~tier1_mask)

        res_tiers[tier0_mask] = 0.05
        res_tiers[tier1_mask] = 0.15
        res_tiers[tier2_mask] = 0.50

        # Ring index: computed by integrating ring thicknesses
        # Quantize range into continuous ring index
        ring_indices = np.zeros(n_points, dtype=np.int32)

        # For points in tier 0: ring = floor(r / 0.05)
        ring_indices[tier0_mask] = np.floor(r[tier0_mask] / 0.05).astype(np.int32)

        # For points in tier 1: ring = 200 + floor((r - 10.0) / 0.15)
        r_t1 = np.maximum(0.0, r[tier1_mask] - 10.0)
        ring_indices[tier1_mask] = 200 + np.floor(r_t1 / 0.15).astype(np.int32)

        # For points in tier 2: ring = 333 + floor((r - 30.0) / 0.50)
        r_t2 = np.maximum(0.0, r[tier2_mask] - 30.0)
        ring_indices[tier2_mask] = 334 + np.floor(r_t2 / 0.50).astype(np.int32)

        # Compute nominal ring center radius for consistent angular binning within each ring
        r_ring = np.zeros(n_points, dtype=np.float32)
        r_ring[tier0_mask] = (ring_indices[tier0_mask] + 0.5) * 0.05
        r_ring[tier1_mask] = 10.0 + (ring_indices[tier1_mask] - 200 + 0.5) * 0.15
        r_ring[tier2_mask] = 30.0 + (ring_indices[tier2_mask] - 334 + 0.5) * 0.50

        # Angular step per ring: dtheta = res_tier / max(r_ring, 0.5)
        clamped_r_ring = np.maximum(r_ring, self.min_range_m)
        dtheta = res_tiers / clamped_r_ring
        
        # Angle index: floor((theta + pi) / dtheta)
        # Shift [-pi, pi] to [0, 2pi]
        theta_shifted = np.mod(theta + np.pi, 2 * np.pi)
        angle_indices = np.floor(theta_shifted / dtheta).astype(np.int32)

        return ring_indices, angle_indices, res_tiers

    def get_cell_spatial_center(
        self, ring_idx: int, angle_idx: int, res_tier: float = 0.05
    ) -> Tuple[float, float, float, float]:
        """
        Calculates center (x, y, r, theta) for a given (ring_idx, angle_idx).
        """
        if ring_idx < 200:
            res = 0.05
            r = (ring_idx + 0.5) * res
        elif ring_idx < 334:
            res = 0.15
            r = 10.0 + (ring_idx - 200 + 0.5) * res
        else:
            res = 0.50
            r = 30.0 + (ring_idx - 334 + 0.5) * res

        dtheta = res / max(r, self.min_range_m)
        theta = (angle_idx + 0.5) * dtheta - np.pi
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return float(x), float(y), float(r), float(theta)

    def aggregate_cell(
        self,
        z_vals: List[float],
        classes: List[float],
        res_tier: float,
        frame_id: int,
    ) -> GridCell:
        """
        Aggregates points in a single cell with ultra-fast pure Python paths
        and multi-layer overhang detection.
        """
        n = len(z_vals)
        if n == 0:
            return GridCell(
                elevation_ground=0.0,
                elevation_obstacle_bottom=None,
                elevation_obstacle_top=None,
                semantic_class=-1,
                point_count=0,
                last_updated_frame=frame_id,
                resolution_tier=res_tier,
            )

        if n == 1:
            z0 = z_vals[0]
            c0 = int(classes[0])
            return GridCell(
                elevation_ground=z0,
                elevation_obstacle_bottom=None,
                elevation_obstacle_top=None,
                semantic_class=c0,
                point_count=1,
                last_updated_frame=frame_id,
                resolution_tier=res_tier,
                confidence=0.4,
            )

        # Multi-point cell
        sorted_z = sorted(z_vals)
        z_min = sorted_z[0]

        # Multi-layer gap detection
        obs_bottom = None
        obs_top = None
        max_gap = 0.0
        gap_split_idx = -1

        for idx in range(n - 1):
            gap = sorted_z[idx + 1] - sorted_z[idx]
            if gap > max_gap:
                max_gap = gap
                gap_split_idx = idx

        if max_gap > self.overhang_gap_threshold_m and gap_split_idx >= 0:
            # Overhang split
            elevation_ground = sorted_z[0]
            obs_bottom = sorted_z[gap_split_idx + 1]
            obs_top = sorted_z[-1]
        else:
            elevation_ground = z_min

        # Majority semantic class
        # Fast count
        class_counts = {}
        for c in classes:
            c_int = int(c)
            if c_int >= 0:
                class_counts[c_int] = class_counts.get(c_int, 0) + 1

        if class_counts:
            best_cls = max(class_counts.items(), key=lambda item: item[1])[0]
        else:
            best_cls = -1

        confidence = 1.0 if n >= self.min_points_per_ring_sector else 0.4

        return GridCell(
            elevation_ground=float(elevation_ground),
            elevation_obstacle_bottom=float(obs_bottom) if obs_bottom is not None else None,
            elevation_obstacle_top=float(obs_top) if obs_top is not None else None,
            semantic_class=int(best_cls),
            point_count=n,
            last_updated_frame=frame_id,
            resolution_tier=res_tier,
            confidence=confidence,
        )

    def project_to_grid(
        self,
        points: PointCloud,
        vehicle_state: Optional[VehicleState] = None,
        frame_id: int = 0,
        roof_height_m: Optional[float] = None,
        use_foveation: bool = True,
    ) -> GridMap:
        """
        Projects (N, 5) point cloud to 2.5D GridMap dictionary using vectorized
        key grouping, bulk single-cell creation, and overhang detection.
        
        Args:
            points: (N, 5) float32 array [x, y, z, intensity, class_id].
            vehicle_state: Current ego vehicle speed and steering.
            frame_id: Frame sequence index.
            roof_height_m: Z-clipping threshold. If None, uses config value.
            use_foveation: Whether to apply dynamic speed/steering fine-zone deform.
            
        Returns:
            GridMap: Dict[(ring_idx, angle_idx), GridCell]
        """
        if len(points) == 0:
            return {}

        roof_h = roof_height_m if roof_height_m is not None else self.roof_height_m

        # Step 1: Z-clip: drop points with z > roof_height_m
        z_mask = points[:, 2] <= roof_h
        valid_points = points[z_mask]
        if len(valid_points) == 0:
            return {}

        x = valid_points[:, 0]
        y = valid_points[:, 1]
        z = valid_points[:, 2]
        classes = valid_points[:, 4]

        # Step 2: Compute range r and azimuth angle theta
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)

        # Range filter: min_range <= r <= max_range
        range_mask = (r >= self.min_range_m) & (r <= self.max_range_m)
        r = r[range_mask]
        theta = theta[range_mask]
        z = z[range_mask]
        classes = classes[range_mask]

        if len(r) == 0:
            return {}

        # Step 3, 4, 5: Compute bin indices
        ring_indices, angle_indices, res_tiers = self.compute_ring_angle_indices(
            r=r,
            theta=theta,
            vehicle_state=vehicle_state,
            use_foveation=use_foveation,
        )

        # 64-bit packed key sorting
        keys = (ring_indices.astype(np.int64) << 32) | (angle_indices.astype(np.int64) & 0xFFFFFFFF)
        sort_idx = np.argsort(keys)
        keys_s = keys[sort_idx]
        z_s = z[sort_idx]
        cls_s = classes[sort_idx]
        res_s = res_tiers[sort_idx]

        unique_keys, starts, counts = np.unique(keys_s, return_index=True, return_counts=True)

        grid_map: GridMap = {}
        single_mask = (counts == 1)
        multi_mask = ~single_mask

        # Step 6a: Bulk single-point cell creation
        s_keys = unique_keys[single_mask]
        s_starts = starts[single_mask]
        s_rings = (s_keys >> 32).tolist()
        s_angles = (s_keys & 0xFFFFFFFF).tolist()
        s_z = z_s[s_starts].tolist()
        s_cls = cls_s[s_starts].astype(np.int32).tolist()
        s_res = res_s[s_starts].tolist()

        for r_i, a_i, z_i, c_i, res_i in zip(s_rings, s_angles, s_z, s_cls, s_res):
            grid_map[(r_i, a_i)] = GridCell(
                elevation_ground=z_i,
                elevation_obstacle_bottom=None,
                elevation_obstacle_top=None,
                semantic_class=c_i,
                point_count=1,
                last_updated_frame=frame_id,
                resolution_tier=res_i,
                confidence=0.4,
            )

        # Step 6b: Multi-point cells with overhang gap detection
        m_indices = np.where(multi_mask)[0]
        gap_thresh = self.overhang_gap_threshold_m
        min_pts = self.min_points_per_ring_sector

        for idx in m_indices:
            uk = unique_keys[idx]
            ring = int(uk >> 32)
            angle = int(uk & 0xFFFFFFFF)
            start = starts[idx]
            cnt = counts[idx]
            res = float(res_s[start])

            sub_z = z_s[start : start + cnt]
            sub_cls = cls_s[start : start + cnt]

            sub_z_sort = np.sort(sub_z)
            z_min = float(sub_z_sort[0])

            gaps = np.diff(sub_z_sort)
            max_gap_idx = int(np.argmax(gaps))
            max_gap = float(gaps[max_gap_idx])

            if max_gap > gap_thresh:
                obs_bottom = float(sub_z_sort[max_gap_idx + 1])
                obs_top = float(sub_z_sort[-1])
            else:
                obs_bottom = None
                obs_top = None

            int_cls = sub_cls.astype(np.int32)
            valid_cls = int_cls[int_cls >= 0]
            best_cls = int(np.bincount(valid_cls).argmax()) if len(valid_cls) > 0 else -1

            grid_map[(ring, angle)] = GridCell(
                elevation_ground=z_min,
                elevation_obstacle_bottom=obs_bottom,
                elevation_obstacle_top=obs_top,
                semantic_class=best_cls,
                point_count=cnt,
                last_updated_frame=frame_id,
                resolution_tier=res,
                confidence=1.0 if cnt >= min_pts else 0.4,
            )

        return grid_map
