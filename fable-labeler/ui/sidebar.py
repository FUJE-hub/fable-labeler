import tkinter as tk

from utils import THEME, FONT_SECTION, FONT_LABEL, FONT_BUTTON, FONT_MONO, make_button


class Sidebar(tk.Frame):
    def __init__(self, master,
                 on_image_selected=None,
                 on_label_added=None,
                 on_label_removed=None,
                 on_label_selected=None,
                 on_export_json=None,
                 on_export_csv=None,
                 on_ann_selected=None):
        super().__init__(master, bg=THEME["bg_dark"], width=260)
        self.on_image_selected = on_image_selected
        self.on_label_added = on_label_added
        self.on_label_removed = on_label_removed
        self.on_label_selected = on_label_selected
        self.on_export_json = on_export_json
        self.on_export_csv = on_export_csv
        self.on_ann_selected = on_ann_selected
        self.pack_propagate(False)

        self._build_project_section()
        self._build_ann_list_section()
        self._build_label_section()
        self._build_stats_section()
        self._build_export_section()

    def _section_label(self, text):
        tk.Label(
            self, text=text, bg=THEME["bg_dark"], fg=THEME["text_primary"],
            font=FONT_SECTION, anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(10, 3))

    def _build_project_section(self):
        self._section_label("图片列表")

        list_frame = tk.Frame(self, bg=THEME["bg_dark"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(
            list_frame, bg=THEME["bg_input"], fg=THEME["text_primary"],
            selectbackground=THEME["accent"], selectforeground="#ffffff",
            font=FONT_MONO, highlightthickness=0, borderwidth=0,
            activestyle="none", yscrollcommand=scrollbar.set,
        )
        self.image_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_select)

    def _build_ann_list_section(self):
        self._section_label("当前图片标注")

        ann_frame = tk.Frame(self, bg=THEME["bg_dark"])
        ann_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        ann_scroll = tk.Scrollbar(ann_frame)
        ann_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.ann_listbox = tk.Listbox(
            ann_frame, bg=THEME["bg_input"], fg=THEME["text_primary"],
            selectbackground=THEME["accent"], selectforeground="#ffffff",
            font=FONT_MONO, highlightthickness=0, borderwidth=0,
            activestyle="none", height=6, yscrollcommand=ann_scroll.set,
        )
        self.ann_listbox.pack(fill=tk.X)
        ann_scroll.config(command=self.ann_listbox.yview)
        self.ann_listbox.bind("<<ListboxSelect>>", self._on_ann_select)

    def _build_stats_section(self):
        self._section_label("项目统计")

        self.stats_label = tk.Label(
            self, text="未加载", bg=THEME["bg_dark"], fg=THEME["text_secondary"],
            font=FONT_MONO, justify=tk.LEFT, anchor="w",
        )
        self.stats_label.pack(fill=tk.X, padx=10, pady=(0, 4))

    def _build_label_section(self):
        self._section_label("标签管理")

        add_frame = tk.Frame(self, bg=THEME["bg_dark"])
        add_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.label_entry = tk.Entry(
            add_frame, bg=THEME["bg_control"], fg=THEME["text_primary"],
            insertbackground=THEME["text_primary"],
            font=FONT_LABEL, highlightthickness=0, borderwidth=0,
        )
        self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.label_entry.bind("<Return>", lambda e: self._add_label())

        add_btn = make_button(
            add_frame, text="+", style="primary", width=3,
            font=FONT_BUTTON, command=self._add_label,
        )
        add_btn.pack(side=tk.LEFT, padx=(4, 0))

        label_list_frame = tk.Frame(self, bg=THEME["bg_dark"])
        label_list_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.label_listbox = tk.Listbox(
            label_list_frame, bg=THEME["bg_input"], fg=THEME["text_primary"],
            selectbackground=THEME["accent"], selectforeground="#ffffff",
            font=FONT_LABEL, highlightthickness=0, borderwidth=0,
            activestyle="none", height=6,
        )
        self.label_listbox.pack(fill=tk.X)
        self.label_listbox.bind("<<ListboxSelect>>", self._on_label_select)

        del_btn = make_button(
            self, text="删除选中标签", style="danger", command=self._remove_label,
        )
        del_btn.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _build_export_section(self):
        self._section_label("导出")

        btn_frame = tk.Frame(self, bg=THEME["bg_dark"])
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        json_btn = make_button(
            btn_frame, text="导出 JSON", style="default",
            command=lambda: self.on_export_json and self.on_export_json(),
        )
        json_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        csv_btn = make_button(
            btn_frame, text="导出 CSV", style="default",
            command=lambda: self.on_export_csv and self.on_export_csv(),
        )
        csv_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

    def set_images(self, files):
        self.image_listbox.delete(0, tk.END)
        for f in files:
            self.image_listbox.insert(tk.END, f)

    def update_image_progress(self, project):
        if not project:
            return
        files = project.get_image_files()
        for i, f in enumerate(files):
            anns = project.get_annotations(f)
            self.image_listbox.itemconfig(
                i, fg=THEME["success"] if anns else THEME["text_primary"]
            )

    def highlight_image(self, index):
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(index)
        self.image_listbox.see(index)

    def set_labels(self, labels):
        self.label_listbox.delete(0, tk.END)
        for lb in labels:
            self.label_listbox.insert(tk.END, lb)

    def get_selected_label(self):
        sel = self.label_listbox.curselection()
        return self.label_listbox.get(sel[0]) if sel else None

    def set_annotations_list(self, annotations):
        lb = self.ann_listbox
        prev_count = lb.size()
        new_count = len(annotations)
        common = min(prev_count, new_count)
        for i in range(common):
            ann = annotations[i]
            prefix = ann.id[:6] if hasattr(ann, 'id') and ann.id else f"{i+1}"
            text = f"[{prefix}] {ann.label}"
            if lb.get(i) != text:
                lb.delete(i)
                lb.insert(i, text)
        if new_count > prev_count:
            for i in range(prev_count, new_count):
                ann = annotations[i]
                prefix = ann.id[:6] if hasattr(ann, 'id') and ann.id else f"{i+1}"
                lb.insert(tk.END, f"[{prefix}] {ann.label}")
        elif new_count < prev_count:
            lb.delete(new_count, tk.END)

    def highlight_ann(self, index):
        self.ann_listbox.selection_clear(0, tk.END)
        if 0 <= index < self.ann_listbox.size():
            self.ann_listbox.selection_set(index)
            self.ann_listbox.see(index)

    def set_stats(self, text):
        self.stats_label.config(text=text)

    def _on_ann_select(self, event):
        sel = self.ann_listbox.curselection()
        if sel and self.on_ann_selected:
            self.on_ann_selected(sel[0])

    def _on_select(self, event):
        sel = self.image_listbox.curselection()
        if sel and self.on_image_selected:
            self.on_image_selected(sel[0])

    def _on_label_select(self, event):
        sel = self.label_listbox.curselection()
        if sel and self.on_label_selected:
            self.on_label_selected(self.label_listbox.get(sel[0]))

    def _add_label(self):
        text = self.label_entry.get().strip()
        if len(text) > 50:
            text = text[:50]
        if text:
            self.label_entry.delete(0, tk.END)
            if self.on_label_added:
                self.on_label_added(text)

    def _remove_label(self):
        sel = self.label_listbox.curselection()
        if sel:
            label = self.label_listbox.get(sel[0])
            if self.on_label_removed:
                self.on_label_removed(label)
