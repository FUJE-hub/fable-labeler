import gc
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

from models.annotation import Annotation, Project
from models.color_extractor import (
    compute_color_stats, validate_annotation, validate_all_annotations,
    generate_corrections, generate_batch_corrections, compute_cross_image_consistency,
)
from models.point_cloud import (
    generate_image_point_cloud, generate_bbox_point_cloud,
    export_point_cloud_csv, export_point_cloud_json,
)
from models.logger import OperationLogger
from models.exporter import export_coco, export_voc, export_yolo
from models import config as cfg_module
from ui.canvas_widget import CanvasWidget
from ui.sidebar import Sidebar
from ui.rgb_panel import RGBPanel
from ui.pointcloud_panel import PointCloudPanel
from utils import (
    THEME, UNDO_STACK_MAX, DEFAULT_WINDOW_SIZE,
    FONT_LABEL, FONT_LABEL_BOLD, FONT_BUTTON,
    FONT_MONO, FONT_STATUS,
    center_window, safe_close_pil, snapshot_annotations, restore_annotations,
    make_button, make_label,
)

PRELOAD_DELTAS = (-2, -1, 1, 2)
PRELOAD_CACHE_MAX = 6
AUTO_SAVE_INTERVAL_MS = 30000
SAMPLE_SIZE_IMAGE = 3000
SAMPLE_SIZE_BBOX = 1500
CALIB_PARAMS = [
    ("z_score_threshold", "Z-Score 阈值", 1.0, 5.0, 0.1),
    ("iqr_multiplier", "IQR 倍数", 0.5, 3.0, 0.1),
    ("mahalanobis_threshold", "马氏距离阈值", 2.0, 10.0, 0.5),
    ("brightness_variance_max", "亮度方差上限", 0.0001, 0.01, 0.0001),
    ("cross_image_cv_max", "跨图CV上限", 0.1, 1.0, 0.05),
]


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Fable 数据标注工具")
        self.root.configure(bg=THEME["bg_dark"])

        self.project = None
        self.current_index = -1
        self.current_image = None
        self.current_point_cloud = None
        self._label_popup = None
        self._label_popup_ann_index = -1
        self._undo_stack = []
        self._redo_stack = []
        self._logger = None
        self._last_used_label = None
        self._preload_cache = {}
        self._cross_image_cache = None
        self._labeled_count = 0
        self._total_anns_count = 0
        self._dirty = False

        self._build_menu()
        self._build_layout()
        self._bind_keys()
        self._center_window()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(AUTO_SAVE_INTERVAL_MS, self._auto_save_tick)

    def _center_window(self):
        w, h = DEFAULT_WINDOW_SIZE.split("x")
        center_window(self.root, int(w), int(h))

    # ── Menu ──────────────────────────────────────────────
    def _build_menu(self):
        menubar = tk.Menu(
            self.root, bg=THEME["bg_dark"], fg=THEME["text_primary"],
            activebackground=THEME["accent"], activeforeground="#ffffff",
            relief="flat",
        )

        file_menu = self._make_menu(menubar)
        file_menu.add_command(label="打开图片目录...", command=self.open_folder, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="保存项目", command=self.save_project, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)

        edit_menu = self._make_menu(menubar)
        edit_menu.add_command(label="撤销", command=self.undo_annotation, accelerator="Ctrl+Z")
        edit_menu.add_command(label="重做", command=self.redo_annotation, accelerator="Ctrl+Y")
        edit_menu.add_command(label="删除选中标注", command=self.delete_selected, accelerator="Delete")
        menubar.add_cascade(label="编辑", menu=edit_menu)

        analyze_menu = self._make_menu(menubar)
        analyze_menu.add_command(label="验证选中标注", command=self.verify_selected)
        analyze_menu.add_command(label="验证当前图片所有标注", command=self.verify_current_all)
        analyze_menu.add_separator()
        analyze_menu.add_command(label="跨图一致性分析", command=self.run_cross_image_analysis)
        analyze_menu.add_separator()
        analyze_menu.add_command(label="生成图片点云", command=self.generate_image_pc)
        analyze_menu.add_command(label="生成选中标注点云", command=self.generate_selected_pc)
        analyze_menu.add_separator()
        analyze_menu.add_command(label="阈值校准...", command=self.open_calibration)
        menubar.add_cascade(label="分析", menu=analyze_menu)

        export_menu = self._make_menu(menubar)
        export_menu.add_command(label="导出标注 JSON", command=self.export_json)
        export_menu.add_command(label="导出标注 CSV", command=self.export_csv)
        export_menu.add_separator()
        export_menu.add_command(label="导出 COCO JSON", command=self.export_coco_format)
        export_menu.add_command(label="导出 VOC XML", command=self.export_voc_format)
        export_menu.add_command(label="导出 YOLO TXT", command=self.export_yolo_format)
        export_menu.add_separator()
        export_menu.add_command(label="导出点云 CSV", command=self.export_pc_csv)
        export_menu.add_command(label="导出点云 JSON", command=self.export_pc_json)
        export_menu.add_separator()
        export_menu.add_command(label="导出操作日志", command=self.export_logs)
        menubar.add_cascade(label="导出", menu=export_menu)

        self.root.config(menu=menubar)

    def _make_menu(self, parent):
        return tk.Menu(
            parent, tearoff=0,
            bg=THEME["bg_input"], fg=THEME["text_primary"],
            activebackground=THEME["accent"], activeforeground="#ffffff",
            relief="flat",
        )

    # ── Layout ────────────────────────────────────────────
    def _build_layout(self):
        self.sidebar = Sidebar(
            self.root,
            on_image_selected=self.on_image_selected,
            on_label_added=self.on_label_added,
            on_label_removed=self.on_label_removed,
            on_label_selected=self.on_label_selected,
            on_export_json=self.export_json,
            on_export_csv=self.export_csv,
            on_ann_selected=self.on_ann_list_selected,
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.rgb_panel = RGBPanel(
            self.root,
            on_verify=self.verify_selected,
            on_verify_all=self.verify_current_all,
            on_cross_image=self.run_cross_image_analysis,
        )
        self.rgb_panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.pc_panel = PointCloudPanel(
            self.root,
            on_export_csv=self.export_pc_csv,
            on_export_json=self.export_pc_json,
        )
        self.pc_panel.pack(side=tk.RIGHT, fill=tk.Y)

        center_frame = tk.Frame(self.root, bg=THEME["bg_dark"])
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_bar = tk.Label(
            center_frame,
            text="请打开图片目录开始标注  |  快捷键: ←→切图, []跳未标注, 1-9改标签, Del删除, Ctrl+Z/Y撤销重做, Home重置视图",
            bg=THEME["bg_panel"], fg=THEME["text_muted"],
            font=FONT_STATUS, anchor="w", padx=10, pady=5,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas_widget = CanvasWidget(
            center_frame,
            on_box_created=self.on_box_created,
            on_box_selected=self.on_box_selected,
            on_box_moved=self.on_box_moved,
            on_box_resized=self.on_box_resized,
            on_drag_start=self.on_drag_start,
        )
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    # ── Key bindings ──────────────────────────────────────
    def _bind_keys(self):
        self.root.bind("<Control-o>", lambda e: self.open_folder())
        self.root.bind("<Control-s>", lambda e: self.save_project())
        self.root.bind("<Control-z>", lambda e: self.undo_annotation())
        self.root.bind("<Control-y>", lambda e: self.redo_annotation())
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<Escape>", lambda e: self._close_label_popup())
        self.root.bind("<Left>", lambda e: self.navigate_image(-1))
        self.root.bind("<Right>", lambda e: self.navigate_image(1))
        self.root.bind("<bracketleft>", lambda e: self.navigate_unlabeled(-1))
        self.root.bind("<bracketright>", lambda e: self.navigate_unlabeled(1))
        self.root.bind("<Home>", lambda e: self._reset_view())

        for i in range(1, 10):
            self._bind_digit(i)

    def _bind_digit(self, i):
        def handler(event):
            focused = self.root.focus_get()
            if isinstance(focused, tk.Entry):
                return
            self._quick_label(i - 1)
        self.root.bind(str(i), handler)

    # ── Project open / image loading ──────────────────────
    def open_folder(self):
        folder = filedialog.askdirectory(title="选择图片目录")
        if not folder:
            return
        self.project = Project(folder)
        self.project.load()
        config_path = os.path.join(folder, ".labeler_config.json")
        if os.path.exists(config_path):
            cfg_module.load_config(config_path)
        files = self.project.get_image_files()
        if not files:
            messagebox.showinfo("提示", "该目录下没有找到图片文件")
            self.project = None
            return
        self.status_bar.config(text=f"正在加载 {len(files)} 张图片...")
        self.root.update_idletasks()
        self._init_logger(folder)
        self.sidebar.set_images(files)
        self.sidebar.set_labels(self.project.labels)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._preload_cache.clear()
        self._cross_image_cache = None
        self._labeled_count = 0
        self._total_anns_count = 0
        self.status_bar.config(text=f"已加载 {len(files)} 张图片")
        self._rebuild_labeled_cache()
        self.load_image_at(0)

    def _init_logger(self, folder):
        self._logger = OperationLogger(os.path.join(folder, ".labeler_logs"))
        self._logger.start_session()

    def _rebuild_labeled_cache(self):
        if not self.project:
            self._labeled_count = 0
            self._total_anns_count = 0
            return
        files = self.project.get_image_files()
        self._labeled_count = 0
        self._total_anns_count = 0
        for f in files:
            count = len(self.project.get_annotations(f))
            self._total_anns_count += count
            if count > 0:
                self._labeled_count += 1
        for i, f in enumerate(files):
            count = len(self.project.get_annotations(f))
            if self.sidebar.image_listbox.size() > i:
                color = THEME["success"] if count > 0 else THEME["text_primary"]
                self.sidebar.image_listbox.itemconfig(i, fg=color)

    def _load_pil_image(self):
        if not self.project or self.current_index < 0:
            return None
        files = self.project.get_image_files()
        image_path = os.path.join(self.project.project_dir, files[self.current_index])
        img = None
        try:
            img = Image.open(image_path)
            img.load()
            return img.convert("RGB")
        except Exception:
            if img is not None:
                safe_close_pil(img)
            return None

    def _release_current_image(self):
        if self.current_image is not None:
            safe_close_pil(self.current_image)
            self.current_image = None
        self.current_point_cloud = None
        gc.collect()

    def _preload_adjacent(self):
        if not self.project:
            return
        files = self.project.get_image_files()
        wanted = set()
        for delta in PRELOAD_DELTAS:
            idx = self.current_index + delta
            if 0 <= idx < len(files):
                wanted.add(idx)
        for idx in list(self._preload_cache):
            if idx not in wanted:
                safe_close_pil(self._preload_cache.pop(idx))
        while len(self._preload_cache) > PRELOAD_CACHE_MAX:
            farthest = max(self._preload_cache, key=lambda k: abs(k - self.current_index))
            safe_close_pil(self._preload_cache.pop(farthest))
        for idx in wanted:
            if idx not in self._preload_cache:
                path = os.path.join(self.project.project_dir, files[idx])
                try:
                    img = Image.open(path)
                    img.load()
                    self._preload_cache[idx] = img
                except Exception:
                    pass

    def _clear_preload_cache(self):
        for img in self._preload_cache.values():
            safe_close_pil(img)
        self._preload_cache.clear()

    def load_image_at(self, index):
        if not self.project:
            return
        files = self.project.get_image_files()
        if not files or index < 0 or index >= len(files):
            return
        self.current_index = index
        image_name = files[index]
        image_path = os.path.join(self.project.project_dir, image_name)
        self._release_current_image()
        try:
            img_copy = self.canvas_widget.load_image(image_path)
        except Exception as e:
            messagebox.showwarning("加载失败", f"无法加载图片 {image_name}:\n{e}")
            return
        anns = self.project.get_annotations(image_name)
        self.canvas_widget.set_annotations(anns)
        self.sidebar.highlight_image(index)
        if index in self._preload_cache:
            cached_img = self._preload_cache.pop(index)
            try:
                self.current_image = cached_img.convert("RGB")
            finally:
                safe_close_pil(cached_img)
            safe_close_pil(img_copy)
        else:
            if img_copy:
                try:
                    self.current_image = img_copy.convert("RGB")
                finally:
                    safe_close_pil(img_copy)
            else:
                self.current_image = self._load_pil_image()
        self.rgb_panel.clear()
        self.pc_panel.clear()
        if self._logger:
            self._logger.log_open_image(image_name)
        self._update_status()
        self._preload_adjacent()

    # ── Navigation ────────────────────────────────────────
    def navigate_image(self, delta):
        files = self.project.get_image_files() if self.project else []
        new_index = self.current_index + delta
        if 0 <= new_index < len(files):
            self.load_image_at(new_index)

    def on_image_selected(self, index):
        self.load_image_at(index)

    def navigate_unlabeled(self, delta):
        if not self.project:
            return
        files = self.project.get_image_files()
        step = 1 if delta > 0 else -1
        i = self.current_index + delta
        while 0 <= i < len(files):
            if not self.project.get_annotations(files[i]):
                self.load_image_at(i)
                return
            i += step
        self.status_bar.config(text="所有图片均已标注")

    def _reset_view(self):
        self.canvas_widget.reset_zoom()

    # ── Box event handlers ────────────────────────────────
    def on_box_created(self, bbox):
        if not self.project:
            return
        selected_label = self._last_used_label or self.sidebar.get_selected_label() or "unlabeled"
        files = self.project.get_image_files()
        image_name = files[self.current_index]
        self._push_undo("create")
        ann = Annotation(bbox=bbox, label=selected_label)
        self.project.add_annotation(image_name, ann)
        self.canvas_widget.set_annotations(self.project.get_annotations(image_name))
        if self._logger:
            idx = len(self.project.get_annotations(image_name)) - 1
            self._logger.log_create_annotation(image_name, selected_label, bbox, idx, ann_id=ann.id)
        self._update_status()
        self._show_label_popup(len(self.project.get_annotations(image_name)) - 1)

    def on_box_selected(self, index):
        self.canvas_widget.select_annotation(index)
        self.sidebar.highlight_ann(index)
        self._update_status()

    def on_drag_start(self, index):
        self._push_undo("drag")

    def on_box_moved(self, index, bbox):
        self.status_bar.config(text=f"已移动标注 {index+1}")

    def on_box_resized(self, index, bbox):
        self.status_bar.config(text=f"已缩放标注 {index+1}")

    # ── Label management callbacks ────────────────────────
    def on_label_added(self, label):
        if self.project:
            self.project.add_label(label)
            self.sidebar.set_labels(self.project.labels)

    def on_label_removed(self, label):
        if self.project:
            self.project.remove_label(label)
            self.sidebar.set_labels(self.project.labels)

    def on_label_selected(self, label):
        pass

    def on_ann_list_selected(self, index):
        if not self.project or self.current_index < 0:
            return
        files = self.project.get_image_files()
        anns = self.project.get_annotations(files[self.current_index])
        if 0 <= index < len(anns):
            self.canvas_widget.select_annotation(index)
            self.sidebar.highlight_ann(index)

    def _quick_label(self, index):
        if not self.project or not self.project.labels or index >= len(self.project.labels):
            return
        label = self.project.labels[index]
        self.sidebar.label_listbox.selection_clear(0, tk.END)
        self.sidebar.label_listbox.selection_set(index)

        idx = self.canvas_widget.selected_index
        if idx >= 0:
            files = self.project.get_image_files()
            anns = self.project.get_annotations(files[self.current_index])
            if idx < len(anns):
                self._push_undo("relabel")
                anns[idx].label = label
                self.canvas_widget.redraw()
                self._last_used_label = label
                self.status_bar.config(text=f"已将标注 {idx+1} 标签改为: {label}")

    # ── Label popup ───────────────────────────────────────
    def _show_label_popup(self, ann_index):
        if not self.project or not self.project.labels:
            return
        self._close_label_popup()

        popup = tk.Toplevel(self.root)
        popup.title("选择标签")
        popup.configure(bg=THEME["bg_input"])
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        self._label_popup = popup
        self._label_popup_ann_index = ann_index

        tk.Label(
            popup, text=f"为标注 {ann_index+1} 选择标签:",
            bg=THEME["bg_input"], fg=THEME["text_primary"],
            font=FONT_LABEL_BOLD,
        ).pack(padx=14, pady=(12, 6))

        for i, label in enumerate(self.project.labels):
            shortcut = f" ({i+1})" if i < 9 else ""
            btn = make_button(
                popup, text=f"{label}{shortcut}", style="default",
                width=22, anchor="w", padx=10, pady=4,
                command=lambda lb=label: self._apply_label_popup(ann_index, lb),
            )
            btn.pack(fill=tk.X, padx=14, pady=2)

        skip_btn = make_button(
            popup, text="跳过 (Esc)", style="ghost",
            command=self._close_label_popup,
        )
        skip_btn.pack(fill=tk.X, padx=14, pady=(8, 12))

        x = self.root.winfo_x() + self.root.winfo_width() // 2 - 130
        y = self.root.winfo_y() + self.root.winfo_height() // 2 - 60
        popup.geometry(f"+{x}+{y}")
        popup.protocol("WM_DELETE_WINDOW", self._close_label_popup)
        popup.bind("<Key>", lambda e: self._on_popup_key(e, ann_index))
        popup.focus_set()

    def _on_popup_key(self, event, ann_index):
        ch = event.char
        if ch.isdigit() and 1 <= int(ch) <= 9:
            idx = int(ch) - 1
            if idx < len(self.project.labels):
                self._apply_label_popup(ann_index, self.project.labels[idx])
            return
        if event.keysym == "Escape":
            self._close_label_popup()

    def _apply_label_popup(self, ann_index, label):
        if not self.project:
            return
        self._last_used_label = label
        files = self.project.get_image_files()
        image_name = files[self.current_index]
        anns = self.project.get_annotations(image_name)
        if ann_index < len(anns):
            self._push_undo("relabel")
            old_label = anns[ann_index].label
            anns[ann_index].label = label
            self.canvas_widget.redraw()
            self.status_bar.config(text=f"已将标注 {ann_index+1} 标签设为: {label}")
            if self._logger:
                self._logger.log_relabel_annotation(
                    image_name, ann_index, old_label, label, ann_id=anns[ann_index].id
                )
        self._close_label_popup()

    def _close_label_popup(self):
        if self._label_popup:
            try:
                self._label_popup.destroy()
            except Exception:
                pass
            self._label_popup = None
        self.canvas_widget.clear_selection()

    # ── Undo / Redo (snapshot = dict list, not Annotation objects) ───
    def _push_undo(self, action):
        if not self.project or self.current_index < 0:
            return
        self._cross_image_cache = None
        files = self.project.get_image_files()
        image_name = files[self.current_index]
        anns = self.project.get_annotations(image_name)
        self._undo_stack.append((image_name, snapshot_annotations(anns)))
        self._redo_stack.clear()
        stack_limit = cfg_module.get("undo_stack_size", UNDO_STACK_MAX)
        if len(self._undo_stack) > stack_limit:
            self._undo_stack.pop(0)

    def undo_annotation(self):
        if not self._undo_stack:
            return
        image_name = self.project.get_image_files()[self.current_index]
        undo_name, undo_snapshot = self._undo_stack.pop()
        cur_anns = self.project.get_annotations(undo_name)
        self._redo_stack.append((undo_name, snapshot_annotations(cur_anns)))
        self.project.set_annotations(undo_name, restore_annotations(undo_snapshot))
        if undo_name == image_name:
            self.canvas_widget.set_annotations(self.project.get_annotations(undo_name))
        self._update_status()

    def redo_annotation(self):
        if not self._redo_stack:
            return
        image_name = self.project.get_image_files()[self.current_index]
        redo_name, redo_snapshot = self._redo_stack.pop()
        cur_anns = self.project.get_annotations(redo_name)
        self._undo_stack.append((redo_name, snapshot_annotations(cur_anns)))
        self.project.set_annotations(redo_name, restore_annotations(redo_snapshot))
        if redo_name == image_name:
            self.canvas_widget.set_annotations(self.project.get_annotations(redo_name))
        self._update_status()

    # ── Delete ────────────────────────────────────────────
    def delete_selected(self):
        if not self.project or self.current_index < 0:
            return
        idx = self.canvas_widget.selected_index
        if idx < 0:
            return
        files = self.project.get_image_files()
        image_name = files[self.current_index]
        self._push_undo("delete")
        ann = self.project.get_annotations(image_name)[idx]
        if self._logger:
            self._logger.log_delete_annotation(image_name, ann.label, ann.bbox, idx, ann_id=ann.id)
        self.project.remove_annotation(image_name, idx)
        self.canvas_widget.set_annotations(self.project.get_annotations(image_name))
        self._update_status()

    # ── Verification ──────────────────────────────────────
    def verify_selected(self):
        if not self.project or self.current_index < 0 or self.current_image is None:
            return
        idx = self.canvas_widget.selected_index
        files = self.project.get_image_files()
        anns = self.project.get_annotations(files[self.current_index])
        if idx < 0 or idx >= len(anns):
            messagebox.showinfo("提示", "请先点击选中一个标注框")
            return
        ann = anns[idx]
        ann.color_stats = compute_color_stats(self.current_image, ann.bbox)
        result = validate_annotation(self.current_image, ann)
        corrections = generate_corrections(self.current_image, ann, anns)
        self.rgb_panel.display_result(ann.color_stats, result, corrections)
        self.status_bar.config(
            text=f"标注 {idx+1} 验证完成: {result['verdict']} (score={result['score']})"
        )

    def verify_current_all(self):
        if not self.project or self.current_index < 0 or self.current_image is None:
            return
        files = self.project.get_image_files()
        image_name = files[self.current_index]
        anns = self.project.get_annotations(image_name)
        if not anns:
            messagebox.showinfo("提示", "当前图片没有标注")
            return
        for ann in anns:
            if not ann.color_stats or "mean_rgb" not in ann.color_stats:
                ann.color_stats = compute_color_stats(self.current_image, ann.bbox)
        report = validate_all_annotations(self.current_image, anns)
        batch_result = generate_batch_corrections(self.project, image_name, self.current_image)
        self.rgb_panel.display_summary(report)
        self.rgb_panel.display_batch_corrections(batch_result)
        self.status_bar.config(
            text=f"验证完成: 通过 {report['pass_count']} / 警告 {report['warn_count']} / "
                 f"失败 {report['fail_count']} / 总计 {report['total']}"
        )

    def run_cross_image_analysis(self):
        if not self.project:
            messagebox.showinfo("提示", "请先打开图片目录")
            return
        files = self.project.get_image_files()
        total_anns = sum(len(self.project.get_annotations(f)) for f in files)
        if total_anns == 0:
            messagebox.showinfo("提示", "项目中没有标注数据")
            return
        verified = sum(
            1 for f in files
            for a in self.project.get_annotations(f)
            if a.color_verified or (a.color_stats and "mean_rgb" in a.color_stats)
        )
        if verified == 0:
            messagebox.showinfo("提示", "请先对标注执行\"验证全部\"以生成颜色特征数据")
            return
        if self._cross_image_cache is None:
            self._cross_image_cache = compute_cross_image_consistency(self.project)
        self.rgb_panel.display_cross_image_result(self._cross_image_cache)
        self.status_bar.config(text=f"跨图一致性分析完成: {len(self._cross_image_cache)} 个标签")

    # ── Calibration ───────────────────────────────────────
    def open_calibration(self):
        win = tk.Toplevel(self.root)
        win.title("阈值校准")
        win.configure(bg=THEME["bg_input"])
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.grab_set()

        vars_map = {}
        for i, (key, label, lo, hi, step) in enumerate(CALIB_PARAMS):
            current = cfg_module.get(key)
            make_label(win, f"{label}:", size="normal").grid(
                row=i, column=0, padx=(14, 6), pady=7, sticky="w"
            )
            var = tk.DoubleVar(value=current)
            entry = tk.Entry(
                win, textvariable=var, width=8,
                bg=THEME["bg_control"], fg=THEME["text_primary"],
                insertbackground=THEME["text_primary"],
                font=FONT_MONO, highlightthickness=0, borderwidth=0,
            )
            entry.grid(row=i, column=1, padx=4, pady=7)
            scale = tk.Scale(
                win, from_=lo, to=hi, resolution=step, orient=tk.HORIZONTAL,
                variable=var, bg=THEME["bg_input"], fg=THEME["text_primary"],
                troughcolor=THEME["bg_control"],
                highlightthickness=0, borderwidth=0, length=200,
                font=("Arial", 8),
            )
            scale.grid(row=i, column=2, padx=(0, 14), pady=7)
            vars_map[key] = var

        def _apply():
            for key, var in vars_map.items():
                cfg_module.set_value(key, var.get())
            self._cross_image_cache = None
            self.status_bar.config(text="阈值已更新")

        def _reset():
            for key, var in vars_map.items():
                var.set(cfg_module.DEFAULTS[key])

        def _preview():
            _apply()
            self.verify_selected()

        def _save_to_file():
            _apply()
            if self.project:
                path = os.path.join(self.project.project_dir, ".labeler_config.json")
                try:
                    cfg_module.save_config(path)
                    self.status_bar.config(text=f"配置已保存: {path}")
                except Exception as e:
                    messagebox.showerror("保存失败", str(e))

        btn_frame = tk.Frame(win, bg=THEME["bg_input"])
        btn_frame.grid(row=len(CALIB_PARAMS), column=0, columnspan=3, pady=(10, 14), padx=14, sticky="ew")

        make_button(btn_frame, "预览效果", style="primary", command=_preview).pack(side=tk.LEFT, padx=(0, 4))
        make_button(btn_frame, "恢复默认", style="ghost", command=_reset).pack(side=tk.LEFT, padx=(0, 4))
        make_button(btn_frame, "保存配置", style="default", command=_save_to_file).pack(side=tk.LEFT, padx=(0, 4))
        make_button(btn_frame, "关闭", style="ghost", command=win.destroy).pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", win.destroy)
        x = self.root.winfo_x() + self.root.winfo_width() // 2 - 250
        y = self.root.winfo_y() + self.root.winfo_height() // 2 - 150
        win.geometry(f"+{x}+{y}")

    # ── Point cloud ───────────────────────────────────────
    def generate_image_pc(self):
        if not self.project or self.current_index < 0 or self.current_image is None:
            return
        self.current_point_cloud = generate_image_point_cloud(self.current_image, SAMPLE_SIZE_IMAGE)
        self.pc_panel.display_point_cloud(self.current_point_cloud, "整图 RGB 点云")
        self.status_bar.config(text=f"已生成整图点云 ({self.current_point_cloud['count']} 个采样点)")

    def generate_selected_pc(self):
        if not self.project or self.current_index < 0 or self.current_image is None:
            return
        idx = self.canvas_widget.selected_index
        if idx < 0:
            messagebox.showinfo("提示", "请先点击选中一个标注框")
            return
        files = self.project.get_image_files()
        anns = self.project.get_annotations(files[self.current_index])
        if idx >= len(anns):
            return
        ann = anns[idx]
        self.current_point_cloud = generate_bbox_point_cloud(self.current_image, ann.bbox, SAMPLE_SIZE_BBOX)
        self.pc_panel.display_point_cloud(self.current_point_cloud, f"标注 {idx+1} ({ann.label}) 点云")
        self.status_bar.config(text=f"已生成标注 {idx+1} 的点云 ({self.current_point_cloud['count']} 个采样点)")

    # ── Save / Export ─────────────────────────────────────
    def save_project(self):
        if not self.project:
            return
        try:
            self.project.save()
            self.status_bar.config(text="项目已保存")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存项目时出错:\n{e}")

    def _ask_save_path(self, ext, filetypes, initialfile=None):
        kwargs = {
            "defaultextension": ext,
            "filetypes": filetypes,
            "initialdir": self.project.project_dir,
        }
        if initialfile:
            kwargs["initialfile"] = initialfile
        return filedialog.asksaveasfilename(**kwargs)

    def export_json(self):
        if not self.project:
            return
        path = self._ask_save_path(".json", [("JSON", "*.json")])
        if path:
            try:
                self.project.export_json(path)
                self.status_bar.config(text=f"已导出 JSON: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 JSON 时出错:\n{e}")

    def export_csv(self):
        if not self.project:
            return
        path = self._ask_save_path(".csv", [("CSV", "*.csv")])
        if path:
            try:
                self.project.export_csv(path)
                self.status_bar.config(text=f"已导出 CSV: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 CSV 时出错:\n{e}")

    def export_coco_format(self):
        if not self.project:
            return
        path = self._ask_save_path(".json", [("JSON", "*.json")], "coco_annotations.json")
        if path:
            try:
                imgs, anns, cats = export_coco(self.project, path)
                msg = f"COCO 导出完成: {imgs} 图 / {anns} 标注 / {cats} 类别"
                self.status_bar.config(text=msg)
                messagebox.showinfo("导出完成", msg)
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 COCO 时出错:\n{e}")

    def _ask_folder(self, title):
        return filedialog.askdirectory(title=title, initialdir=self.project.project_dir)

    def export_voc_format(self):
        if not self.project:
            return
        folder = self._ask_folder("选择 VOC XML 输出目录")
        if folder:
            try:
                count = export_voc(self.project, folder)
                msg = f"VOC 导出完成: {count} 个 XML 文件"
                self.status_bar.config(text=msg)
                messagebox.showinfo("导出完成", msg)
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 VOC 时出错:\n{e}")

    def export_yolo_format(self):
        if not self.project:
            return
        folder = self._ask_folder("选择 YOLO TXT 输出目录")
        if folder:
            try:
                files, anns = export_yolo(self.project, folder)
                msg = f"YOLO 导出完成: {files} 文件 / {anns} 标注"
                self.status_bar.config(text=msg)
                messagebox.showinfo("导出完成", msg)
            except Exception as e:
                messagebox.showerror("导出失败", f"导出 YOLO 时出错:\n{e}")

    def export_pc_csv(self):
        if self.current_point_cloud is None:
            messagebox.showinfo("提示", "请先生成点云")
            return
        path = self._ask_save_path(".csv", [("CSV", "*.csv")])
        if path:
            try:
                export_point_cloud_csv(path, self.current_point_cloud)
                self.status_bar.config(text=f"已导出点云 CSV: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出点云 CSV 时出错:\n{e}")

    def export_pc_json(self):
        if self.current_point_cloud is None:
            messagebox.showinfo("提示", "请先生成点云")
            return
        path = self._ask_save_path(".json", [("JSON", "*.json")])
        if path:
            try:
                export_point_cloud_json(path, self.current_point_cloud)
                self.status_bar.config(text=f"已导出点云 JSON: {path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出点云 JSON 时出错:\n{e}")

    def export_logs(self):
        if not self._logger:
            messagebox.showinfo("提示", "请先打开图片目录")
            return
        self._logger.end_session()
        path = self._logger.save()
        self._logger.start_session()
        self.status_bar.config(text=f"操作日志已导出: {path}")
        messagebox.showinfo("导出完成", f"日志已保存到:\n{path}")

    # ── Status bar ────────────────────────────────────────
    def _update_status(self):
        if not self.project or self.current_index < 0:
            return
        files = self.project.get_image_files()
        image_name = files[self.current_index]
        anns = self.project.get_annotations(image_name)
        total = len(files)
        current_count = len(anns)

        self._sync_labeled_count(files, image_name, current_count)
        self._update_sidebar_color(image_name, current_count)

        self._prev_image_name = image_name
        self._prev_ann_count = current_count
        self._dirty = True

        self.status_bar.config(
            text=f"[{self.current_index + 1}/{total}] {image_name}  |  "
                 f"当前图片: {current_count} 个标注  |  "
                 f"已标注: {self._labeled_count}/{total} 张  |  "
                 f"撤销栈: {len(self._undo_stack)}  |  "
                 f"缩放: {self.canvas_widget._zoom:.1f}x"
        )
        self.sidebar.set_annotations_list(anns)
        if self.canvas_widget.selected_index >= 0:
            self.sidebar.highlight_ann(self.canvas_widget.selected_index)
        self.sidebar.set_stats(
            f"图片总数: {total}\n"
            f"已标注图片: {self._labeled_count}\n"
            f"总标注数: {self._total_anns_count}\n"
            f"标签类别: {len(self.project.labels)}"
        )

    def _sync_labeled_count(self, files, image_name, current_count):
        prev_label = getattr(self, "_prev_image_name", None)
        prev_count = getattr(self, "_prev_ann_count", -1)

        if prev_label is not None and prev_label != image_name:
            old_count = len(self.project.get_annotations(prev_label))
            self._total_anns_count += old_count - prev_count
            was_labeled = prev_count > 0
            new_labeled = old_count > 0
            if was_labeled and not new_labeled:
                self._labeled_count -= 1
            elif not was_labeled and new_labeled:
                self._labeled_count += 1
            self._refresh_listbox_color(files, prev_label, old_count)

        self._total_anns_count += current_count - (prev_count if prev_count >= 0 and prev_label == image_name else current_count)
        if prev_label == image_name:
            if current_count > 0 and prev_count == 0:
                self._labeled_count += 1
            elif current_count == 0 and prev_count > 0:
                self._labeled_count -= 1

    def _refresh_listbox_color(self, files, target_name, count):
        for i, f in enumerate(files):
            if f == target_name:
                color = THEME["success"] if count > 0 else THEME["text_primary"]
                if self.sidebar.image_listbox.size() > i:
                    self.sidebar.image_listbox.itemconfig(i, fg=color)
                break

    def _update_sidebar_color(self, image_name, current_count):
        idx = self.current_index
        if idx < self.sidebar.image_listbox.size():
            color = THEME["success"] if current_count > 0 else THEME["text_primary"]
            self.sidebar.image_listbox.itemconfig(idx, fg=color)

    # ── Cleanup ───────────────────────────────────────────
    def _auto_save_tick(self):
        if self._dirty and self.project:
            try:
                self.project.save()
                self._dirty = False
            except Exception:
                pass
        self.root.after(AUTO_SAVE_INTERVAL_MS, self._auto_save_tick)

    def _on_close(self):
        self._release_current_image()
        self._clear_preload_cache()
        self._undo_stack.clear()
        self._redo_stack.clear()
        if self._logger:
            self._logger.end_session()
            self._logger.save()
        self.root.destroy()
