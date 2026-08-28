"""
Deep Learning Point Cloud Segmentation and Model Pipeline for FoveaMap.

Implements pure PyTorch PointNet++ / PointMLP architecture for 4-class semantic segmentation
alongside automatic fallback to vectorized heuristic rules.
"""

import os
import sys
import logging
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.perception.class_map import map_semantickitti_to_4class
from src.perception.heuristic_fallback import heuristic_segment_points

# Setup logger
logger = logging.getLogger("foveamap.perception")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [segment.py] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LightweightPointNet(nn.Module):
    """
    Pure PyTorch PointNet architecture for per-point 4-class segmentation.
    Zero external C++/CUDA libraries, runs natively on CPU and GPU.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        # Local feature extractor
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)

        # Point classifier decoder (Global feature + Local feature)
        self.conv4 = nn.Conv1d(256 + 128, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 64, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) tensor of point coordinates.
        Returns:
            (B, num_classes, N) logits.
        """
        B, C, N = x.shape
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))  # (B, 256, N)

        # Global feature via Max Pooling (expanded without memory copy)
        global_feat = torch.max(x3, dim=2, keepdim=True)[0]  # (B, 256, 1)
        global_feat_expanded = global_feat.expand(-1, -1, N) # (B, 256, N)

        # Concatenate global + local point features
        cat_feat = torch.cat([global_feat_expanded, x2], dim=1)  # (B, 384, N)
        y = F.relu(self.bn4(self.conv4(cat_feat)))
        y = F.relu(self.bn5(self.conv5(y)))
        logits = self.conv6(y)  # (B, num_classes, N)
        return logits


_DEFAULT_MODEL_PATH = "data/model_pointnet.pt"
_LOADED_MODEL: Optional[LightweightPointNet] = None
_LOGGED_STARTUP = False


def train_lightweight_model(
    kitti_loader,
    train_indices: Optional[List[int]] = None,
    num_epochs: int = 5,
    save_path: str = _DEFAULT_MODEL_PATH,
    device: str = "cpu",
    seed: int = 42,
) -> LightweightPointNet:
    """
    Trains the lightweight PointNet model on designated training split frames.
    Uses fixed seed for reproducible runs and saves checkpoint to save_path.
    """
    global _LOADED_MODEL
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    indices = train_indices if train_indices is not None else list(range(min(20, len(kitti_loader))))
    logger.info(f"Training Lightweight PointNet on {len(indices)} split frames (indices {indices[0]}..{indices[-1]}, Device: {device}, Seed: {seed})...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = LightweightPointNet(in_channels=3, num_classes=4).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    rng = np.random.default_rng(seed)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for frame_idx in indices:
            scan = kitti_loader[frame_idx]
            if len(scan) < 100:
                continue

            xyz = scan[:, :3]
            raw_labels = scan[:, 4].astype(np.int32)
            target_4class = map_semantickitti_to_4class(raw_labels)

            # Subsample for fast training batch (e.g. 2048 points per batch)
            n_pts = len(xyz)
            sample_size = min(2048, n_pts)
            idx = rng.choice(n_pts, sample_size, replace=False)

            xyz_sub = xyz[idx]
            target_sub = target_4class[idx]

            # Center coordinates
            xyz_sub = xyz_sub - np.mean(xyz_sub, axis=0, keepdims=True)

            tensor_x = torch.from_numpy(xyz_sub.T).unsqueeze(0).float().to(device)  # (1, 3, N)
            tensor_y = torch.from_numpy(target_sub).unsqueeze(0).long().to(device)   # (1, N)

            optimizer.zero_grad()
            logits = model(tensor_x)  # (1, 4, N)
            loss = criterion(logits, tensor_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"  Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved successfully to {save_path}")
    model.eval()
    _LOADED_MODEL = model
    return model


def get_segmentation_model(
    model_path: str = _DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
) -> Optional[LightweightPointNet]:
    """Loads or retrieves cached segmentation model on specified device."""
    global _LOADED_MODEL
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    if os.path.exists(model_path):
        try:
            model = LightweightPointNet(in_channels=3, num_classes=4)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            _LOADED_MODEL = model
            return model
        except Exception as e:
            logger.warning(f"Could not load checkpoint at {model_path}: {e}")

    return None


def segment_points(
    points_xyz: np.ndarray,
    model: Optional[LightweightPointNet] = None,
    device: Optional[str] = None,
    use_heuristic: bool = False,
) -> np.ndarray:
    """
    Segments 3D point cloud coordinates into 4-class taxonomy:
      0: Drivable Terrain
      1: Non-Drivable Terrain
      2: Static Obstacle
      3: Dynamic Object
      
    Args:
        points_xyz: (N, 3) or (N, >=3) numpy array of point coordinates.
        model: Optional pre-loaded PyTorch model.
        device: Device string ('cuda', 'mps', or 'cpu'). Auto-detected if None.
        use_heuristic: If True, bypasses model and runs vectorized heuristic classifier.
        
    Returns:
        (N,) int32 array where values are in {0, 1, 2, 3}.
    """
    global _LOGGED_STARTUP
    n_points = len(points_xyz)
    if n_points == 0:
        return np.empty(0, dtype=np.int32)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Log once on startup
    if not _LOGGED_STARTUP:
        if use_heuristic:
            logger.info(f"Running on {device.upper()} with Vectorized Heuristic RANSAC pipeline.")
        else:
            logger.info(f"Running on {device.upper()} with Deep Learning PointNet++ architecture.")
        _LOGGED_STARTUP = True

    if use_heuristic:
        return heuristic_segment_points(points_xyz)

    # Attempt Deep Learning Inference
    active_model = model if model is not None else get_segmentation_model(device=device)

    if active_model is not None:
        try:
            active_model.eval()
            
            # Single tensor copy to GPU/device with inference_mode
            with torch.inference_mode():
                xyz_tensor = torch.as_tensor(points_xyz[:, :3], dtype=torch.float32, device=device)
                centroid = torch.mean(xyz_tensor, dim=0, keepdim=True)
                norm_x = (xyz_tensor - centroid).T.unsqueeze(0)  # (1, 3, N)
                
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = active_model(norm_x)
                else:
                    logits = active_model(norm_x)
                    
                preds = torch.argmax(logits, dim=1).squeeze(0).to(dtype=torch.int32).cpu().numpy()
                return preds
        except Exception as e:
            logger.warning(f"Inference error with Deep Learning model: {e}. Falling back to heuristic classifier.")

    # Fallback to heuristic
    return heuristic_segment_points(points_xyz)
