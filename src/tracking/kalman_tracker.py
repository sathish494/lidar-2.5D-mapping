"""
Kalman Filter Object Tracking & Active Footprint Erasure (Anti-Ghosting).

Implements 2D constant-velocity Kalman filtering for dynamic obstacle tracking (Class 3)
and active footprint erasure to eliminate stale/ghost occupancy trails behind moving objects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np

from src.grid.grid_types import GridCell, GridMap


@dataclass
class TrackedObject:
    """
    State of a tracked dynamic object in world/sensor coordinates.
    """
    track_id: int
    position_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    predicted_next_xy: Tuple[float, float]
    class_id: int
    frames_since_seen: int
    bbox_size_xy: Tuple[float, float] = (3.5, 1.8)  # Length, width in meters
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32) * 0.1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "position_xy": [float(self.position_xy[0]), float(self.position_xy[1])],
            "velocity_xy": [float(self.velocity_xy[0]), float(self.velocity_xy[1])],
            "predicted_next_xy": [float(self.predicted_next_xy[0]), float(self.predicted_next_xy[1])],
            "class_id": self.class_id,
            "frames_since_seen": self.frames_since_seen,
            "bbox_size_xy": [float(self.bbox_size_xy[0]), float(self.bbox_size_xy[1])],
        }


def cluster_dynamic_detections(
    cell_centers: List[Tuple[float, float, int]],
    cluster_dist_m: float = 2.5,
) -> List[Tuple[float, float, int]]:
    """
    Clusters neighboring dynamic cells into unified object centroids (vehicles / pedestrians)
    using vectorized adjacency and connected component graph analysis.
    """
    if not cell_centers:
        return []

    n = len(cell_centers)
    if n == 1:
        return [(cell_centers[0][0], cell_centers[0][1], cell_centers[0][2])]

    pts = np.array([[x, y] for x, y, _ in cell_centers], dtype=np.float32)
    classes = np.array([c for _, _, c in cell_centers], dtype=np.int32)

    # Vectorized distance matrix
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    adj_matrix = (dx * dx + dy * dy) <= (cluster_dist_m * cluster_dist_m)

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        n_components, labels = connected_components(
            csgraph=csr_matrix(adj_matrix), directed=False, return_labels=True
        )

        clusters: List[Tuple[float, float, int]] = []
        for c_id in range(n_components):
            mask = (labels == c_id)
            c_pts = pts[mask]
            centroid_x = float(np.mean(c_pts[:, 0]))
            centroid_y = float(np.mean(c_pts[:, 1]))
            cls_val = int(classes[mask][0])
            clusters.append((centroid_x, centroid_y, cls_val))

        return clusters
    except Exception:
        # Fallback loop
        clusters = []
        visited = set()
        for i in range(n):
            if i in visited:
                continue
            c_indices = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if adj_matrix[i, j]:
                    c_indices.append(j)
                    visited.add(j)
            c_pts = pts[c_indices]
            clusters.append((float(np.mean(c_pts[:, 0])), float(np.mean(c_pts[:, 1])), int(classes[c_indices[0]])))
        return clusters


class KalmanTrackerManager:
    """
    Manages multi-object tracking using 2D Constant-Velocity Kalman Filters.
    """
    def __init__(
        self,
        max_coast_frames: int = 5,
        distance_gate_m: float = 2.5,
        dt_s: float = 0.1,
    ):
        self.max_coast_frames = max_coast_frames
        self.distance_gate_m = distance_gate_m
        self.dt_s = dt_s
        self.next_track_id = 1
        self.tracks: List[TrackedObject] = []
        self.prev_footprints: Dict[int, Set[Tuple[int, int]]] = {}

        # Process and Measurement Noise matrices
        q_pos = 0.05
        q_vel = 0.5
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        self.R = (np.eye(2, dtype=np.float32) * 0.1).astype(np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    def _predict_track(self, track: TrackedObject) -> Tuple[np.ndarray, np.ndarray]:
        """Predict state and covariance forward by dt."""
        F = np.array([
            [1.0, 0.0, self.dt_s, 0.0],
            [0.0, 1.0, 0.0, self.dt_s],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        x = np.array([
            track.position_xy[0],
            track.position_xy[1],
            track.velocity_xy[0],
            track.velocity_xy[1],
        ], dtype=np.float32)

        x_pred = F @ x
        P_pred = F @ track.covariance @ F.T + self.Q
        return x_pred, P_pred

    def update_tracks(
        self,
        detections_xy: List[Tuple[float, float, int]],  # [(x, y, class_id), ...]
    ) -> List[TrackedObject]:
        """
        Associates detections with existing tracks and updates Kalman filters.
        """
        # Step 1: Predict all existing tracks
        predicted_states = []
        for t in self.tracks:
            x_pred, P_pred = self._predict_track(t)
            predicted_states.append((x_pred, P_pred))
            t.predicted_next_xy = (float(x_pred[0]), float(x_pred[1]))

        # Step 2: Greedy Euclidean Association with distance gating
        unmatched_detections = set(range(len(detections_xy)))
        matched_tracks = set()

        if len(self.tracks) > 0 and len(detections_xy) > 0:
            # Build cost matrix
            cost_matrix = np.zeros((len(self.tracks), len(detections_xy)), dtype=np.float32)
            for i, (x_pred, _) in enumerate(predicted_states):
                for j, (det_x, det_y, _) in enumerate(detections_xy):
                    dist = np.hypot(x_pred[0] - det_x, x_pred[1] - det_y)
                    cost_matrix[i, j] = dist

            # Match nearest within distance gate
            for i in range(len(self.tracks)):
                min_j = int(np.argmin(cost_matrix[i]))
                min_dist = cost_matrix[i, min_j]
                if min_dist <= self.distance_gate_m and min_j in unmatched_detections:
                    # Match found
                    matched_tracks.add(i)
                    unmatched_detections.remove(min_j)

                    det_x, det_y, det_cls = detections_xy[min_j]
                    z_meas = np.array([det_x, det_y], dtype=np.float32)
                    x_pred, P_pred = predicted_states[i]

                    # Kalman Update
                    y_res = z_meas - self.H @ x_pred
                    S = self.H @ P_pred @ self.H.T + self.R
                    K = P_pred @ self.H.T @ np.linalg.inv(S)
                    x_updated = x_pred + K @ y_res
                    P_updated = (np.eye(4, dtype=np.float32) - K @ self.H) @ P_pred

                    # Update track
                    t = self.tracks[i]
                    t.position_xy = (float(x_updated[0]), float(x_updated[1]))
                    t.velocity_xy = (float(x_updated[2]), float(x_updated[3]))
                    t.class_id = det_cls
                    t.frames_since_seen = 0
                    t.covariance = P_updated

        # Step 3: Coast unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                x_pred, P_pred = predicted_states[i]
                t.position_xy = (float(x_pred[0]), float(x_pred[1]))
                t.velocity_xy = (float(x_pred[2]), float(x_pred[3]))
                t.frames_since_seen += 1
                t.covariance = P_pred

        # Step 4: Spawn new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det_x, det_y, det_cls = detections_xy[det_idx]
            new_track = TrackedObject(
                track_id=self.next_track_id,
                position_xy=(float(det_x), float(det_y)),
                velocity_xy=(0.0, 0.0),
                predicted_next_xy=(float(det_x), float(det_y)),
                class_id=det_cls,
                frames_since_seen=0,
                covariance=np.diag([0.1, 0.1, 5.0, 5.0]).astype(np.float32),
            )
            self.next_track_id += 1
            self.tracks.append(new_track)

        # Step 5: Prune dead tracks exceeding max_coast_frames
        self.tracks = [t for t in self.tracks if t.frames_since_seen <= self.max_coast_frames]
        return self.tracks


def erase_vacated_footprints(
    grid_map: GridMap,
    tracks: List[TrackedObject],
    prev_footprints: Dict[int, Set[Tuple[int, int]]],
    frame_id: int,
    grid_engine=None,
) -> Tuple[GridMap, Dict[int, Set[Tuple[int, int]]], int]:
    """
    Vectorized Active Footprint Erasure (Anti-Ghosting):
    Immediately eliminates stale dynamic occupancy (Class 3) in cells vacated by moving tracks
    using vectorized broadcast distance filtering with zero Python loops over cells x tracks.
    Never touches static obstacles (Class 2) or terrain (Class 0, 1).
    
    Returns:
        updated_grid: Cleaned GridMap.
        updated_footprints: New footprint map for next frame.
        erased_count: Number of ghost cells actively erased.
    """
    curr_footprints: Dict[int, Set[Tuple[int, int]]] = {t.track_id: set() for t in tracks}
    erased_count = 0

    if not tracks:
        return grid_map, curr_footprints, 0

    # Step 1: Extract dynamic items (Class 3 only)
    dynamic_items = [(k, c) for k, c in grid_map.items() if c.semantic_class == 3]
    if not dynamic_items:
        # If no current dynamic cells exist, verify previous footprints directly
        for t in tracks:
            tid = t.track_id
            for v_key in prev_footprints.get(tid, set()):
                if v_key in grid_map and grid_map[v_key].semantic_class == 3:
                    c = grid_map[v_key]
                    c.semantic_class = 0
                    c.confidence = 0.8
                    erased_count += 1
        return grid_map, curr_footprints, erased_count

    # Step 2: Vectorized polar coordinate reconstruction
    keys = [k for k, _ in dynamic_items]
    rings = np.array([k[0] for k in keys], dtype=np.float32)
    angles = np.array([k[1] for k in keys], dtype=np.float32)

    # Resolution tiers: <200 -> 0.05m, 200..333 -> 0.15m, >=334 -> 0.50m
    t0_m = rings < 200
    t1_m = (rings >= 200) & (rings < 334)
    t2_m = rings >= 334

    r_arr = np.empty_like(rings)
    res_arr = np.empty_like(rings)
    r_arr[t0_m] = (rings[t0_m] + 0.5) * 0.05
    res_arr[t0_m] = 0.05
    r_arr[t1_m] = 10.0 + (rings[t1_m] - 200 + 0.5) * 0.15
    res_arr[t1_m] = 0.15
    r_arr[t2_m] = 30.0 + (rings[t2_m] - 334 + 0.5) * 0.50
    res_arr[t2_m] = 0.50

    dtheta = res_arr / np.maximum(r_arr, 0.5)
    theta = (angles + 0.5) * dtheta - np.pi
    cx = r_arr * np.cos(theta)
    cy = r_arr * np.sin(theta)

    # Step 3: Broadcast pairwise distance check against all active tracks
    tx = np.array([t.position_xy[0] for t in tracks], dtype=np.float32)
    ty = np.array([t.position_xy[1] for t in tracks], dtype=np.float32)
    tr = np.array([max(t.bbox_size_xy) * 0.8 for t in tracks], dtype=np.float32)

    dx = cx[:, None] - tx[None, :]
    dy = cy[:, None] - ty[None, :]
    in_track = (dx * dx + dy * dy) <= (tr * tr)[None, :]

    for t_idx, t in enumerate(tracks):
        matched_indices = np.where(in_track[:, t_idx])[0]
        curr_footprints[t.track_id] = {keys[i] for i in matched_indices}

    # Step 4: Compare against previous footprints and actively erase vacated cells
    for t in tracks:
        tid = t.track_id
        prev_keys = prev_footprints.get(tid, set())
        curr_keys = curr_footprints.get(tid, set())

        vacated_keys = prev_keys - curr_keys
        for v_key in vacated_keys:
            if v_key in grid_map:
                cell = grid_map[v_key]
                # STRICT GUARD: Only erase cells that are marked Dynamic Object (Class 3)
                if cell.semantic_class == 3:
                    cell.semantic_class = 0
                    cell.elevation_obstacle_bottom = None
                    cell.elevation_obstacle_top = None
                    cell.confidence = 0.8
                    erased_count += 1

    return grid_map, curr_footprints, erased_count
