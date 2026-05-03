import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import THEME, FONT_SECTION, FONT_LABEL, FONT_BUTTON, FONT_MONO, make_button

COLOR_MAP = {"PASS": THEME["pass_color"], "WARN": THEME["warn_color"], "FAIL": THEME["fail_color"]}
SEVERITY_MAP = {"HIGH": THEME["fail_color"], "MEDIUM": THEME["warn_color"], "NONE": THEME["pass_color"]}


class RGBPanel(tk.Frame):
    def __init__(self, master, on_verify=None, on_verify_all=None, on_cross_image=None):
        super().__init__(master, bg=THEME["bg_dark"], width=340)
        self.on_verify = on_verify
        self.on_verify_all = on_verify_all
        self.on_cross_image = on_cross_image
        self.pack_propagate(False)
        self._build()

    def _build(self):
        tk.Label(
            self, text="RGB 验证", bg=THEME["bg_dark"], fg=THEME["text_primary"],
            font=FONT_SECTION, anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(10, 4))

        btn_frame = tk.Frame(self, bg=THEME["bg_dark"])
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        make_button(
            btn_frame, text="验证选中", style="primary",
            command=lambda: self.on_verify and self.on_verify(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        make_button(
            btn_frame, text="验证全部", style="default",
            command=lambda: self.on_verify_all and self.on_verify_all(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        make_button(
            self, text="跨图一致性分析", style="primary",
            command=lambda: self.on_cross_image and self.on_cross_image(),
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

        self.info_frame = tk.Frame(self, bg=THEME["bg_panel"])
        self.info_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.verdict_label = tk.Label(
            self.info_frame, text="未验证", bg=THEME["bg_panel"], fg=THEME["text_muted"],
            font=(THEME["font_family"], 14, "bold"),
        )
        self.verdict_label.pack(anchor="w", padx=10, pady=(8, 0))

        self.score_label = tk.Label(
            self.info_frame, text="", bg=THEME["bg_panel"], fg=THEME["text_secondary"],
            font=FONT_MONO,
        )
        self.score_label.pack(anchor="w", padx=10, pady=(2, 0))

        self.stats_label = tk.Label(
            self.info_frame, text="", bg=THEME["bg_panel"], fg=THEME["text_secondary"],
            font=FONT_MONO, justify=tk.LEFT, wraplength=310,
        )
        self.stats_label.pack(anchor="w", padx=10, pady=(2, 8))

        tk.Label(
            self, text="颜色直方图", bg=THEME["bg_dark"], fg=THEME["text_secondary"],
            font=FONT_LABEL, anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 2))

        self.hist_figure = Figure(figsize=(3.3, 1.5), dpi=80, facecolor=THEME["bg_input"])
        self.hist_axes = self.hist_figure.add_subplot(111)
        self.hist_axes.set_facecolor(THEME["bg_input"])
        self.hist_axes.tick_params(colors=THEME["text_muted"], labelsize=7)
        for spine in self.hist_axes.spines.values():
            spine.set_color(THEME["border"])
        self.hist_figure.tight_layout(pad=1.5)

        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, master=self)
        self.hist_canvas.get_tk_widget().pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(
            self, text="主色", bg=THEME["bg_dark"], fg=THEME["text_secondary"],
            font=FONT_LABEL, anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 2))

        self.color_bar_frame = tk.Frame(self, bg=THEME["bg_input"], height=22)
        self.color_bar_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(
            self, text="异常检测 & 纠错建议", bg=THEME["bg_dark"], fg=THEME["text_secondary"],
            font=FONT_LABEL, anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 2))

        self.issues_text = tk.Text(
            self, bg=THEME["bg_input"], fg=THEME["text_primary"], font=FONT_MONO,
            height=8, highlightthickness=0, borderwidth=0, wrap=tk.WORD, state=tk.DISABLED,
        )
        self.issues_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def display_result(self, stats, validation, corrections=None):
        verdict = validation["verdict"]
        score = validation["score"]
        issues = validation["issues"]

        color = COLOR_MAP.get(verdict, THEME["text_muted"])
        self.verdict_label.config(text=verdict, fg=color)

        pixel_count = stats.get("pixel_count", 0)
        mean_rgb = stats.get("mean_rgb", [0, 0, 0])
        std_rgb = stats.get("std_rgb", [0, 0, 0])
        self.score_label.config(text=f"Score: {score:.2f}")
        self.stats_label.config(
            text=f"像素数: {pixel_count}\n"
                 f"均值 RGB: ({mean_rgb[0]:.0f}, {mean_rgb[1]:.0f}, {mean_rgb[2]:.0f})\n"
                 f"标准差:   ({std_rgb[0]:.0f}, {std_rgb[1]:.0f}, {std_rgb[2]:.0f})\n"
                 f"亮度: {stats.get('brightness_mean', 0):.3f}  "
                 f"饱和度: {stats.get('saturation_mean', 0):.3f}"
        )

        self._draw_histogram(stats.get("histogram", {}))
        self._draw_color_bars(stats.get("dominant_colors", []))
        self._set_issues_and_corrections(issues, corrections)

    def display_summary(self, report):
        total = report["total"]
        self.verdict_label.config(text="验证报告", fg=THEME["text_primary"])

        anomaly = report.get("anomaly_detection", {})
        anomaly_info = ""
        if anomaly and anomaly.get("method") != "insufficient_data":
            anomaly_info = (f"\n异常检测: {anomaly.get('anomaly_count', 0)} 异常 / "
                          f"{anomaly.get('suspicious_count', 0)} 可疑 / "
                          f"{anomaly.get('normal_count', 0)} 正常")

        self.score_label.config(text=f"平均分: {report['avg_score']:.2f}")
        self.stats_label.config(
            text=f"总计: {total} 个标注\n"
                 f"通过: {report['pass_count']}  警告: {report['warn_count']}  "
                 f"失败: {report['fail_count']}{anomaly_info}"
        )

    def display_cross_image_result(self, consistency):
        self.verdict_label.config(text="跨图一致性", fg="#4fc3f7")
        if not consistency:
            self.score_label.config(text="无数据")
            self.stats_label.config(text="请先对所有图片执行\"验证全部\"")
            self._set_text("")
            return

        lines = []
        for label, info in consistency.items():
            score = info.get("consistency_score")
            status = info.get("status", "")
            count = info.get("count", 0)
            images = info.get("images_count", 0)
            anomalies = info.get("anomaly_count", 0)
            if score is not None:
                lines.append(f"[{label}] {status} (score={score:.2f})")
                lines.append(f"  {count}个标注 / {images}张图 / {anomalies}异常")
            else:
                lines.append(f"[{label}] {status} ({count}个)")

        self.score_label.config(text=f"共 {len(consistency)} 个标签")
        self.stats_label.config(text="")
        self._set_text("\n".join(lines))

    def display_batch_corrections(self, batch_result):
        self.verdict_label.config(text="纠错建议", fg=THEME["warning"])

        high = batch_result.get("high_count", 0)
        medium = batch_result.get("medium_count", 0)
        total = batch_result.get("total_annotations", 0)

        self.score_label.config(text=f"HIGH: {high}  MEDIUM: {medium}")
        self.stats_label.config(text=f"共 {total} 个标注")

        lines = []
        for i, corrections in enumerate(batch_result.get("per_annotation", [])):
            for c in corrections:
                if c["severity"] == "NONE":
                    continue
                sev = c["severity"]
                marker = "!!!" if sev == "HIGH" else "!  "
                lines.append(f"[{marker}] 标注{i+1}: {c['message']}")
        if not lines:
            lines.append("所有标注未发现明显问题")

        self._set_text("\n".join(lines))

    def _set_issues_and_corrections(self, issues, corrections=None):
        self.issues_text.config(state=tk.NORMAL)
        self.issues_text.delete("1.0", tk.END)

        content_parts = []
        if issues:
            for i, issue in enumerate(issues, 1):
                content_parts.append(f"{i}. {issue}")
        if corrections:
            for c in corrections:
                sev = c.get("severity", "NONE")
                if sev == "NONE":
                    continue
                prefix = "[!]" if sev == "HIGH" else "[~]"
                content_parts.append(f"{prefix} {c['message']}")
        if not content_parts:
            content_parts.append("无问题")

        self.issues_text.insert(tk.END, "\n".join(content_parts))
        self.issues_text.config(state=tk.DISABLED)

    def _set_text(self, text):
        self.issues_text.config(state=tk.NORMAL)
        self.issues_text.delete("1.0", tk.END)
        self.issues_text.insert(tk.END, text or "无数据")
        self.issues_text.config(state=tk.DISABLED)

    def _draw_histogram(self, histogram):
        self.hist_axes.clear()
        self.hist_axes.set_facecolor(THEME["bg_input"])

        if not histogram:
            self.hist_canvas.draw()
            return

        bins = list(range(8))
        bin_centers = [i * 32 + 16 for i in bins]
        width = 28

        h_r = histogram.get("r", [0] * 8)
        h_g = histogram.get("g", [0] * 8)
        h_b = histogram.get("b", [0] * 8)

        max_val = max(max(h_r, default=1), max(h_g, default=1), max(h_b, default=1), 1)

        self.hist_axes.bar([x - width for x in bin_centers], [v / max_val for v in h_r],
                          width=width, color="#ff4444", alpha=0.7, label="R")
        self.hist_axes.bar(bin_centers, [v / max_val for v in h_g],
                          width=width, color="#44ff44", alpha=0.7, label="G")
        self.hist_axes.bar([x + width for x in bin_centers], [v / max_val for v in h_b],
                          width=width, color="#4488ff", alpha=0.7, label="B")

        self.hist_axes.set_xlim(0, 256)
        self.hist_axes.set_ylim(0, 1.1)
        self.hist_axes.set_xticks([0, 64, 128, 192, 256])
        self.hist_axes.tick_params(colors=THEME["text_muted"], labelsize=7)
        self.hist_axes.legend(
            fontsize=7, loc="upper right",
            facecolor=THEME["bg_input"], edgecolor=THEME["border"], labelcolor=THEME["text_primary"],
        )
        for spine in self.hist_axes.spines.values():
            spine.set_color(THEME["border"])
        self.hist_canvas.draw()

    def _draw_color_bars(self, dominants):
        for widget in self.color_bar_frame.winfo_children():
            widget.destroy()
        if not dominants:
            return
        total_ratio = sum(d["ratio"] for d in dominants)
        for d in dominants:
            ratio = d["ratio"] / total_ratio
            hex_color = d["hex"]
            bar = tk.Frame(self.color_bar_frame, bg=hex_color, height=22)
            bar.pack(side=tk.LEFT, fill=tk.Y, expand=False,
                    ipadx=max(2, int(300 * ratio)))

    def clear(self):
        self.verdict_label.config(text="未验证", fg=THEME["text_muted"])
        self.score_label.config(text="")
        self.stats_label.config(text="")
        self._set_text("")
        self.hist_axes.clear()
        self.hist_axes.set_facecolor(THEME["bg_input"])
        for spine in self.hist_axes.spines.values():
            spine.set_color(THEME["border"])
        self.hist_canvas.draw()
        for widget in self.color_bar_frame.winfo_children():
            widget.destroy()
