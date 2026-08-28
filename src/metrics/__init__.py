"""
FoveaMap Evaluation & Metrics Subsystem.
"""

from src.metrics.evaluate import (
    compute_miou,
    evaluate_segmentation_by_distance,
    benchmark_latency,
    benchmark_memory_savings,
    run_full_evaluation,
)

__all__ = [
    "compute_miou",
    "evaluate_segmentation_by_distance",
    "benchmark_latency",
    "benchmark_memory_savings",
    "run_full_evaluation",
]
