"""
SemanticKITTI Ingestion Module for FoveaMap.

Loads binary point clouds (.bin) and ground-truth semantic labels (.label).
Formats scans into unified (N, 5) float32 arrays: [x, y, z, intensity, class_id].
"""

import os
import glob
from typing import Optional, List, Tuple
import numpy as np


def load_velodyne_bin(bin_path: str) -> np.ndarray:
    """
    Load a raw Velodyne binary point cloud file.
    
    Args:
        bin_path: Path to .bin file.
        
    Returns:
        (N, 4) float32 array: [x, y, z, intensity]
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Velodyne binary file not found: {bin_path}")
    
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Corrupted binary file (size {points.size} not divisible by 4): {bin_path}")
    
    return points.reshape(-1, 4)


def load_labels(label_path: str) -> np.ndarray:
    """
    Load a SemanticKITTI .label file.
    
    Format: 32-bit unsigned integers where:
      - Lower 16 bits: semantic class ID (label & 0xFFFF)
      - Upper 16 bits: instance ID (label >> 16)
      
    Args:
        label_path: Path to .label file.
        
    Returns:
        (N,) int32 array: raw semantic class IDs.
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_labels = (raw_labels & 0xFFFF).astype(np.int32)
    return semantic_labels


def load_kitti_scan(bin_path: str, label_path: Optional[str] = None) -> np.ndarray:
    """
    Load a single KITTI scan and optional ground-truth label, returning an (N, 5) array.
    
    Args:
        bin_path: Path to .bin file.
        label_path: Optional path to .label file.
        
    Returns:
        (N, 5) float32 array: [x, y, z, intensity, class_id].
        class_id is -1 if label_path is None or missing.
    """
    points = load_velodyne_bin(bin_path)
    n_points = points.shape[0]
    
    scan = np.zeros((n_points, 5), dtype=np.float32)
    scan[:, :4] = points
    
    if label_path is not None and os.path.exists(label_path):
        labels = load_labels(label_path)
        if labels.shape[0] != n_points:
            raise ValueError(
                f"Mismatch between points count ({n_points}) and labels count ({labels.shape[0]})"
            )
        scan[:, 4] = labels.astype(np.float32)
    else:
        scan[:, 4] = -1.0  # Unclassified
        
    return scan


class KITTILoader:
    """
    Dataset loader for a sequence of SemanticKITTI / KITTI-format binary scans.
    Includes synthetic signature detection guard to prevent misidentifying procedural stand-ins as real sensor captures.
    """
    def __init__(self, dataset_dir: str = "data/synthetic_kitti_like", warn_on_synthetic: bool = True):
        # Fallback if old path passed
        if not os.path.exists(dataset_dir) and dataset_dir == "data/kitti_sample" and os.path.exists("data/synthetic_kitti_like"):
            dataset_dir = "data/synthetic_kitti_like"
        elif not os.path.exists(dataset_dir) and dataset_dir == "data/synthetic_kitti_like" and os.path.exists("data/kitti_sample"):
            dataset_dir = "data/kitti_sample"

        self.dataset_dir = dataset_dir
        self.velodyne_dir = os.path.join(dataset_dir, "velodyne")
        self.labels_dir = os.path.join(dataset_dir, "labels")
        self.warn_on_synthetic = warn_on_synthetic
        
        self.bin_files: List[str] = []
        self.label_files: List[Optional[str]] = []
        self.is_synthetic_signature: bool = False
        self._discover_files()

    def _discover_files(self) -> None:
        if not os.path.exists(self.velodyne_dir):
            return
        
        self.bin_files = sorted(glob.glob(os.path.join(self.velodyne_dir, "*.bin")))
        self.label_files = []
        
        for bin_f in self.bin_files:
            basename = os.path.splitext(os.path.basename(bin_f))[0]
            label_f = os.path.join(self.labels_dir, f"{basename}.label")
            if os.path.exists(label_f):
                self.label_files.append(label_f)
            else:
                self.label_files.append(None)

        # Loud-fail / warning guard for synthetic signatures
        self._check_synthetic_signature()

    def _check_synthetic_signature(self) -> None:
        """
        Inspects file sizes. Genuine LiDAR captures have variable beam returns per rotation.
        Uniform byte sizes across all frames indicate procedurally generated synthetic stand-ins.
        """
        if len(self.bin_files) >= 2:
            sizes = [os.path.getsize(f) for f in self.bin_files]
            if len(set(sizes)) == 1:
                self.is_synthetic_signature = True
                if self.warn_on_synthetic:
                    print(
                        f"\n[WARNING] [KITTILoader] SYNTHETIC DATA SIGNATURE DETECTED in '{self.dataset_dir}':\n"
                        f"  All {len(self.bin_files)} scans are byte-identical in size ({sizes[0]:,} bytes / {sizes[0]//16:,} points).\n"
                        f"  This directory contains procedurally generated stand-ins, NOT genuine physical Velodyne sensor captures.\n",
                        flush=True,
                    )

    def __len__(self) -> int:
        return len(self.bin_files)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.get_scan(idx)

    def get_scan(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.bin_files):
            raise IndexError(f"Scan index {idx} out of range (0 to {len(self.bin_files)-1})")
        return load_kitti_scan(self.bin_files[idx], self.label_files[idx])
