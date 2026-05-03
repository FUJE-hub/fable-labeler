from __future__ import annotations

import json
import os
import uuid
from typing import Any

from utils import SUPPORTED_IMAGE_EXTS


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Annotation:
    def __init__(
        self,
        bbox: tuple,
        label: str,
        attributes: dict[str, Any] | None = None,
        color_stats: dict[str, Any] | None = None,
        color_verified: bool = False,
        ann_id: str | None = None,
    ) -> None:
        self.id = ann_id or _new_id()
        self.bbox = bbox
        self.label = label
        self.attributes = attributes or {}
        self.color_stats = color_stats or {}
        self.color_verified = color_verified

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "bbox": list(self.bbox),
            "label": self.label,
            "attributes": self.attributes,
        }
        if self.color_stats:
            result["color_stats"] = self.color_stats
        if self.color_verified:
            result["color_verified"] = self.color_verified
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        return cls(
            bbox=data["bbox"],
            label=data["label"],
            attributes=data.get("attributes", {}),
            color_stats=data.get("color_stats", {}),
            color_verified=data.get("color_verified", False),
            ann_id=data.get("id"),
        )

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return abs(x2 - x1) * abs(y2 - y1)


class Project:
    VERSION: int = 2

    def __init__(self, project_dir: str) -> None:
        self.project_dir = project_dir
        self.labels: list[str] = []
        self.image_annotations: dict[str, list[Annotation]] = {}
        self.meta_path = os.path.join(project_dir, ".labeler_meta.json")
        self._file_cache: list[str] | None = None
        self._version: int = 0

    def add_label(self, label: str) -> None:
        if label not in self.labels:
            self.labels.append(label)

    def remove_label(self, label: str) -> None:
        if label in self.labels:
            self.labels.remove(label)

    def get_annotations(self, image_name: str) -> list[Annotation]:
        return self.image_annotations.get(image_name, [])

    def set_annotations(self, image_name: str, annotations: list[Annotation]) -> None:
        self.image_annotations[image_name] = annotations

    def add_annotation(self, image_name: str, annotation: Annotation) -> None:
        if image_name not in self.image_annotations:
            self.image_annotations[image_name] = []
        self.image_annotations[image_name].append(annotation)

    def remove_annotation(self, image_name: str, index: int) -> None:
        anns = self.image_annotations.get(image_name, [])
        if 0 <= index < len(anns):
            anns.pop(index)

    def get_image_files(self) -> list[str]:
        if self._file_cache is not None:
            return self._file_cache
        files: list[str] = []
        for f in os.listdir(self.project_dir):
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS:
                files.append(f)
        files.sort()
        self._file_cache = files
        return files

    def invalidate_cache(self) -> None:
        self._file_cache = None

    def save(self) -> None:
        data: dict[str, Any] = {
            "version": self.VERSION,
            "labels": self.labels,
            "image_annotations": {
                name: [a.to_dict() for a in anns]
                for name, anns in self.image_annotations.items()
            },
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.meta_path):
            self._version = self.VERSION
            return
        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._version = data.get("version", 1)
        self.labels = data.get("labels", [])
        self.image_annotations = {}
        for name, anns in data.get("image_annotations", {}).items():
            self.image_annotations[name] = [Annotation.from_dict(a) for a in anns]

    def export_json(self, output_path: str) -> None:
        result: list[dict[str, Any]] = []
        for name, anns in self.image_annotations.items():
            result.append({
                "image": name,
                "annotations": [a.to_dict() for a in anns],
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def export_csv(self, output_path: str) -> None:
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "annotation_id", "label", "x_min", "y_min", "x_max", "y_max", "attributes"])
            for name, anns in self.image_annotations.items():
                for a in anns:
                    x1, y1, x2, y2 = a.bbox
                    writer.writerow([
                        name, a.id, a.label,
                        f"{x1:.4f}", f"{y1:.4f}", f"{x2:.4f}", f"{y2:.4f}",
                        json.dumps(a.attributes, ensure_ascii=False),
                    ])
