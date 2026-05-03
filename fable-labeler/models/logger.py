from __future__ import annotations

import time
import json
import os
from typing import Any
from datetime import datetime


class OperationLogger:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._events: list[dict[str, Any]] = []
        self._session_start: float | None = None
        self._image_enter_time: float | None = None
        self._current_image: str | None = None

    def start_session(self) -> None:
        self._session_start = time.time()
        self._events = []
        self._log("SESSION_START", {})

    def end_session(self) -> None:
        self._log_image_leave()
        self._log("SESSION_END", {"duration_sec": round(time.time() - self._session_start, 2) if self._session_start else 0})

    def log_open_image(self, image_name: str) -> None:
        if self._current_image:
            self._log_image_leave()
        self._current_image = image_name
        self._image_enter_time = time.time()
        self._log("IMAGE_OPEN", {"image": image_name})

    def log_create_annotation(self, image_name: str, label: str, bbox: tuple, index: int, *, ann_id: str = "") -> None:
        d: dict[str, Any] = {
            "image": image_name,
            "label": label,
            "bbox": [round(v, 4) for v in bbox],
            "index": index,
        }
        if ann_id:
            d["ann_id"] = ann_id
        self._log("ANNOTATION_CREATE", d)

    def log_delete_annotation(self, image_name: str, label: str, bbox: tuple, index: int, *, ann_id: str = "") -> None:
        d: dict[str, Any] = {
            "image": image_name,
            "label": label,
            "bbox": [round(v, 4) for v in bbox],
            "index": index,
        }
        if ann_id:
            d["ann_id"] = ann_id
        self._log("ANNOTATION_DELETE", d)

    def log_move_annotation(self, image_name: str, index: int, old_bbox: tuple, new_bbox: tuple, *, ann_id: str = "") -> None:
        d: dict[str, Any] = {
            "image": image_name,
            "index": index,
            "old_bbox": [round(v, 4) for v in old_bbox],
            "new_bbox": [round(v, 4) for v in new_bbox],
        }
        if ann_id:
            d["ann_id"] = ann_id
        self._log("ANNOTATION_MOVE", d)

    def log_resize_annotation(self, image_name: str, index: int, old_bbox: tuple, new_bbox: tuple, *, ann_id: str = "") -> None:
        d: dict[str, Any] = {
            "image": image_name,
            "index": index,
            "old_bbox": [round(v, 4) for v in old_bbox],
            "new_bbox": [round(v, 4) for v in new_bbox],
        }
        if ann_id:
            d["ann_id"] = ann_id
        self._log("ANNOTATION_RESIZE", d)

    def log_relabel_annotation(self, image_name: str, index: int, old_label: str, new_label: str, *, ann_id: str = "") -> None:
        d: dict[str, Any] = {
            "image": image_name,
            "index": index,
            "old_label": old_label,
            "new_label": new_label,
        }
        if ann_id:
            d["ann_id"] = ann_id
        self._log("ANNOTATION_RELABEL", d)

    def log_verify(self, image_name: str, annotation_count: int, result_summary: str) -> None:
        self._log("VERIFY", {
            "image": image_name,
            "annotation_count": annotation_count,
            "result": result_summary,
        })

    def log_export(self, format_type: str, path: str, annotation_count: int) -> None:
        self._log("EXPORT", {
            "format": format_type,
            "path": path,
            "annotation_count": annotation_count,
        })

    def log_undo(self, action: str) -> None:
        self._log("UNDO", {"action": action})

    def log_redo(self, action: str) -> None:
        self._log("REDO", {"action": action})

    def save(self) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"log_{ts}.json")

        stats = self._compute_stats()
        data: dict[str, Any] = {
            "session_start": self._session_start,
            "session_end": time.time(),
            "events": self._events,
            "statistics": stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _compute_stats(self) -> dict[str, Any]:
        total_time = 0
        if self._session_start:
            total_time = round(time.time() - self._session_start, 2)

        creates = [e for e in self._events if e["type"] == "ANNOTATION_CREATE"]
        deletes = [e for e in self._events if e["type"] == "ANNOTATION_DELETE"]
        moves = [e for e in self._events if e["type"] == "ANNOTATION_MOVE"]
        resizes = [e for e in self._events if e["type"] == "ANNOTATION_RESIZE"]
        relabels = [e for e in self._events if e["type"] == "ANNOTATION_RELABEL"]
        verifies = [e for e in self._events if e["type"] == "VERIFY"]
        image_opens = [e for e in self._events if e["type"] == "IMAGE_OPEN"]
        exports = [e for e in self._events if e["type"] == "EXPORT"]

        label_counts: dict[str, int] = {}
        for e in creates:
            lb = e["data"].get("label", "unknown")
            label_counts[lb] = label_counts.get(lb, 0) + 1

        image_times: dict[str, dict[str, Any]] = {}
        for e in self._events:
            if e["type"] == "IMAGE_OPEN":
                img = e["data"]["image"]
                image_times[img] = {"enter": e["timestamp"], "leave": None, "duration": 0}
            elif e["type"] == "IMAGE_LEAVE":
                img = e["data"]["image"]
                if img in image_times:
                    image_times[img]["leave"] = e["timestamp"]
                    image_times[img]["duration"] = e["data"]["duration_sec"]

        return {
            "total_time_sec": total_time,
            "images_opened": len(image_opens),
            "annotations_created": len(creates),
            "annotations_deleted": len(deletes),
            "annotations_moved": len(moves),
            "annotations_resized": len(resizes),
            "annotations_relabeled": len(relabels),
            "verifications": len(verifies),
            "exports": len(exports),
            "label_distribution": label_counts,
            "per_image_time": {k: v["duration"] for k, v in image_times.items()},
            "avg_time_per_image": round(
                sum(v["duration"] for v in image_times.values()) / max(len(image_times), 1), 2
            ),
            "avg_time_per_annotation": round(
                total_time / max(len(creates), 1), 2
            ),
        }

    def _log_image_leave(self) -> None:
        if self._current_image and self._image_enter_time:
            dur = round(time.time() - self._image_enter_time, 2)
            self._log("IMAGE_LEAVE", {"image": self._current_image, "duration_sec": dur})
            self._image_enter_time = None
            self._current_image = None

    def _log(self, event_type: str, data: dict[str, Any]) -> None:
        self._events.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        })
