from __future__ import annotations

import os
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING, Callable

from models.exporter import export_coco, export_voc, export_yolo, _validate_project
from models.point_cloud import export_point_cloud_csv, export_point_cloud_json

if TYPE_CHECKING:
    from models.annotation import Project


class ExportController:
    def __init__(self, get_project: Callable[[], Project | None],
                 on_status: Callable[[str], None]) -> None:
        self._get_project = get_project
        self._on_status = on_status

    def _ask_save_path(self, ext: str, filetypes: list, initialfile: str | None = None) -> str:
        project = self._get_project()
        if not project:
            return ""
        kwargs: dict = {
            "defaultextension": ext,
            "filetypes": filetypes,
            "initialdir": project.project_dir,
        }
        if initialfile:
            kwargs["initialfile"] = initialfile
        return filedialog.asksaveasfilename(**kwargs)

    def _ask_folder(self, title: str) -> str:
        project = self._get_project()
        if not project:
            return ""
        return filedialog.askdirectory(title=title, initialdir=project.project_dir)

    def _validate_before_export(self) -> bool:
        project = self._get_project()
        if not project:
            return False
        valid, errors = _validate_project(project)
        if not valid:
            error_msg = "\n".join(errors[:10])
            if len(errors) > 10:
                error_msg += f"\n... 还有 {len(errors) - 10} 个错误"
            messagebox.showwarning("验证失败", f"发现无效标注:\n{error_msg}\n\n请修复后再导出")
            return False
        return True

    def export_json(self) -> None:
        project = self._get_project()
        if not project:
            return
        if not self._validate_before_export():
            return
        path = self._ask_save_path(".json", [("JSON", "*.json")])
        if path:
            try:
                project.export_json(path)
                self._on_status(f"已导出 JSON: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 JSON 时出错:\n{e}")

    def export_csv(self) -> None:
        project = self._get_project()
        if not project:
            return
        if not self._validate_before_export():
            return
        path = self._ask_save_path(".csv", [("CSV", "*.csv")])
        if path:
            try:
                project.export_csv(path)
                self._on_status(f"已导出 CSV: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 CSV 时出错:\n{e}")

    def export_coco_format(self) -> None:
        project = self._get_project()
        if not project:
            return
        if not self._validate_before_export():
            return
        path = self._ask_save_path(".json", [("JSON", "*.json")], "coco_annotations.json")
        if path:
            try:
                imgs, anns, cats = export_coco(project, path)
                msg = f"COCO 导出完成: {imgs} 图 / {anns} 标注 / {cats} 类别"
                self._on_status(msg)
                messagebox.showinfo("导出完成", msg)
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 COCO 时出错:\n{e}")

    def export_voc_format(self) -> None:
        project = self._get_project()
        if not project:
            return
        if not self._validate_before_export():
            return
        folder = self._ask_folder("选择 VOC XML 输出目录")
        if folder:
            try:
                count = export_voc(project, folder)
                msg = f"VOC 导出完成: {count} 个 XML 文件"
                self._on_status(msg)
                messagebox.showinfo("导出完成", msg)
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 VOC 时出错:\n{e}")

    def export_yolo_format(self) -> None:
        project = self._get_project()
        if not project:
            return
        if not self._validate_before_export():
            return
        folder = self._ask_folder("选择 YOLO TXT 输出目录")
        if folder:
            try:
                files, anns = export_yolo(project, folder)
                msg = f"YOLO 导出完成: {files} 文件 / {anns} 标注"
                self._on_status(msg)
                messagebox.showinfo("导出完成", msg)
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 YOLO 时出错:\n{e}")

    def export_pc_csv(self, current_point_cloud: dict | None) -> None:
        if current_point_cloud is None:
            messagebox.showinfo("提示", "请先生成点云")
            return
        path = self._ask_save_path(".csv", [("CSV", "*.csv")])
        if path:
            try:
                export_point_cloud_csv(path, current_point_cloud)
                self._on_status(f"已导出点云 CSV: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出点云 CSV 时出错:\n{e}")

    def export_pc_json(self, current_point_cloud: dict | None) -> None:
        if current_point_cloud is None:
            messagebox.showinfo("提示", "请先生成点云")
            return
        path = self._ask_save_path(".json", [("JSON", "*.json")])
        if path:
            try:
                export_point_cloud_json(path, current_point_cloud)
                self._on_status(f"已导出点云 JSON: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出点云 JSON 时出错:\n{e}")

    def export_logs(self, logger) -> None:
        if not logger:
            messagebox.showinfo("提示", "请先打开图片目录")
            return
        logger.end_session()
        path = logger.save()
        logger.start_session()
        self._on_status(f"操作日志已导出: {path}")
        messagebox.showinfo("导出完成", f"日志已保存到:\n{path}")
