from __future__ import annotations

from typing import TYPE_CHECKING

from utils import UNDO_STACK_MAX, snapshot_annotations, restore_annotations
from models import config as cfg_module

if TYPE_CHECKING:
    from models.annotation import Project


class UndoManager:
    def __init__(self) -> None:
        self._undo_stack: list[tuple[str, list[dict]]] = []
        self._redo_stack: list[tuple[str, list[dict]]] = []

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_depth(self) -> int:
        return len(self._redo_stack)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def push(self, project: Project, image_name: str) -> None:
        anns = project.get_annotations(image_name)
        self._undo_stack.append((image_name, snapshot_annotations(anns)))
        self._redo_stack.clear()
        stack_limit = cfg_module.get("undo_stack_size", UNDO_STACK_MAX)
        if len(self._undo_stack) > stack_limit:
            self._undo_stack.pop(0)

    def undo(self, project: Project, current_image: str) -> str | None:
        if not self._undo_stack:
            return None
        undo_name, undo_snapshot = self._undo_stack.pop()
        cur_anns = project.get_annotations(undo_name)
        self._redo_stack.append((undo_name, snapshot_annotations(cur_anns)))
        project.set_annotations(undo_name, restore_annotations(undo_snapshot))
        return undo_name

    def redo(self, project: Project, current_image: str) -> str | None:
        if not self._redo_stack:
            return None
        redo_name, redo_snapshot = self._redo_stack.pop()
        cur_anns = project.get_annotations(redo_name)
        self._undo_stack.append((redo_name, snapshot_annotations(cur_anns)))
        project.set_annotations(redo_name, restore_annotations(redo_snapshot))
        return redo_name
