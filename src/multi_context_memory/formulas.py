from __future__ import annotations

import math
from typing import Iterable


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def weighted_score(values: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = 0.0
    total = 0.0
    for key, weight in weights.items():
        total += values.get(key, 0.0) * weight
        total_weight += abs(weight)
    if total_weight == 0:
        return 0.0
    return clamp(total / total_weight)


def vector_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    squared = sum((left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in keys)
    return math.sqrt(squared / len(keys))


def feature_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(abs(left.get(key, 0.0)), abs(right.get(key, 0.0))) for key in keys)
    denominator = sum(max(abs(left.get(key, 0.0)), abs(right.get(key, 0.0))) for key in keys)
    if denominator == 0:
        return 0.0
    return clamp(numerator / denominator)


def token_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return clamp(len(left_set & right_set) / len(left_set | right_set))


def compute_boundary_pressure(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return weighted_score(metrics, weights)


def compute_write_pressure(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return weighted_score(metrics, weights)


def compute_recall_trigger(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return weighted_score(metrics, weights)


def decay_value(value: float, decay_rate: float, dt: float = 1.0) -> float:
    return clamp(value * max(0.0, 1.0 - decay_rate * dt))


def cosineish_salience(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return clamp(sum(values) / len(values))
