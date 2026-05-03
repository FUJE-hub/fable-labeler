from __future__ import annotations

import numpy as np
from PIL import Image


DEFAULT_SAMPLE_COUNT = 3000
BBOX_SAMPLE_COUNT = 1500


def sample_image_pixels(image: Image.Image, n_samples: int = DEFAULT_SAMPLE_COUNT) -> np.ndarray:
    arr = np.array(image).reshape(-1, 3)
    total = len(arr)
    if total == 0:
        return np.array([], dtype=np.uint8).reshape(0, 3)
    if total <= n_samples:
        return arr
    indices = np.random.choice(total, size=n_samples, replace=False)
    return arr[indices]


def sample_bbox_pixels(image: Image.Image, bbox: tuple, n_samples: int = BBOX_SAMPLE_COUNT) -> np.ndarray:
    from models.color_extractor import extract_bbox_pixels
    pixels = extract_bbox_pixels(image, bbox)
    if len(pixels) == 0:
        return np.array([], dtype=np.uint8).reshape(0, 3)
    if len(pixels) <= n_samples:
        return pixels
    indices = np.random.choice(len(pixels), size=n_samples, replace=False)
    return pixels[indices]


def pixels_to_point_cloud(pixels: np.ndarray) -> dict[str, object]:
    if len(pixels) == 0:
        return {"points": [], "colors": [], "count": 0}
    points = pixels.astype(float).tolist()
    colors_norm = (pixels.astype(float) / 255.0).tolist()
    return {
        "points": points,
        "colors": colors_norm,
        "count": len(points),
    }


def generate_image_point_cloud(image: Image.Image, n_samples: int = DEFAULT_SAMPLE_COUNT) -> dict:
    pixels = sample_image_pixels(image, n_samples)
    return pixels_to_point_cloud(pixels)


def generate_bbox_point_cloud(image: Image.Image, bbox: tuple, n_samples: int = BBOX_SAMPLE_COUNT) -> dict:
    pixels = sample_bbox_pixels(image, bbox, n_samples)
    return pixels_to_point_cloud(pixels)


def export_point_cloud_csv(path: str, point_cloud: dict):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "g", "b"])
        for pt in point_cloud["points"]:
            writer.writerow([int(pt[0]), int(pt[1]), int(pt[2])])


def export_point_cloud_json(path: str, point_cloud: dict):
    import json
    data = {
        "count": point_cloud["count"],
        "points": point_cloud["points"],
        "colors": point_cloud["colors"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
