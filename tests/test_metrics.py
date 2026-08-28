"""
Unit tests for Metrics, mIoU Evaluation, and Memory Benchmarks (Milestone 14 and 15).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.evaluate import (
    compute_miou,
    evaluate_segmentation_by_distance,
    benchmark_memory_savings,
)
from src.ingestion.kitti_loader import KITTILoader


def test_compute_miou_perfect_and_partial():
    """Validates mIoU calculation with perfect and partial overlap."""
    gt = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    # Perfect predictions
    pred_perfect = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    miou_p, class_ious_p = compute_miou(gt, pred_perfect, num_classes=4)
    assert miou_p == pytest.approx(1.0)
    assert all(iou == pytest.approx(1.0) for iou in class_ious_p.values())

    # Partial predictions: 1 mistake in class 0 and 1 in class 1
    pred_partial = np.array([0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    miou_part, class_ious_part = compute_miou(gt, pred_partial, num_classes=4)
    # Class 0: TP=1, FP=0, FN=1 -> IoU = 1/2 = 0.5
    # Class 1: TP=2, FP=1, FN=0 -> IoU = 2/3 = 0.6667
    # Class 2: TP=2, FP=0, FN=0 -> IoU = 1.0
    # Class 3: TP=2, FP=0, FN=0 -> IoU = 1.0
    assert class_ious_part[0] == pytest.approx(0.5)
    assert class_ious_part[1] == pytest.approx(2.0 / 3.0)
    assert class_ious_part[2] == pytest.approx(1.0)
    assert class_ious_part[3] == pytest.approx(1.0)
    assert miou_part == pytest.approx((0.5 + 2.0 / 3.0 + 1.0 + 1.0) / 4.0)


def test_memory_savings_benchmark_ge_60_percent():
    """
    Validates Milestone 15 pass criteria:
    Foveated grid vs uniform 5cm grid shows >= 60% memory reduction.
    """
    loader = KITTILoader("data/synthetic_kitti_like")
    assert len(loader) > 0, "KITTI sample data missing"

    mem_report = benchmark_memory_savings(loader)
    mean_savings = mem_report["mean_savings_pct"]

    print(f"\n[BENCHMARK] Mean Memory Savings: {mean_savings}%")
    assert mean_savings >= 60.0, f"Memory savings {mean_savings}% was less than 60.0% requirement"


def test_segmentation_distance_bucket_evaluation():
    """
    Validates Milestone 14 pass criteria:
    mIoU computed across distance buckets (0-10m, 10-30m, 30-100m) for all 4 classes.
    """
    loader = KITTILoader("data/synthetic_kitti_like")
    seg_report = evaluate_segmentation_by_distance(loader, use_heuristic=True, num_frames=5)

    assert "0-10m" in seg_report
    assert "10-30m" in seg_report
    assert "30-100m" in seg_report
    assert "Overall (0-100m)" in seg_report

    # Overall mIoU should be positive
    overall_miou = seg_report["Overall (0-100m)"]["mIoU"]
    assert overall_miou > 50.0, f"Overall mIoU was {overall_miou}%, expected > 50%"
