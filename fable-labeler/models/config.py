from __future__ import annotations

import json
import os
from typing import Any

DEFAULTS: dict[str, Any] = {
    "z_score_threshold": 2.5,
    "iqr_multiplier": 1.5,
    "mahalanobis_threshold": 5.0,
    "min_bbox_pixels": 20,
    "max_bbox_ratio": 0.95,
    "color_coverage_min": 0.8,
    "brightness_variance_max": 0.001,
    "cross_image_cv_max": 0.3,
    "undo_stack_size": 50,
    "point_cloud_sample_size": 3000,
    "point_cloud_bbox_sample_size": 1500,
}

_config: dict[str, Any] = dict(DEFAULTS)


def get(key: str, default: Any = None) -> Any:
    return _config.get(key, default)


def set_value(key: str, value: Any) -> None:
    _config[key] = value


def get_all() -> dict[str, Any]:
    return dict(_config)


def save_config(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_config, f, ensure_ascii=False, indent=2)


def load_config(path: str) -> None:
    global _config
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _config.update(data)


def reset() -> None:
    global _config
    _config = dict(DEFAULTS)
