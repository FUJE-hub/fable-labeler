import tkinter as tk
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import THEME, FONT_SECTION, FONT_BUTTON, FONT_MONO, make_button

PC_DISPLAY_SAMPLE_MAX = 2000


class PointCloudPanel(tk.Frame):
    def __init__(self, master, on_export_csv=None, on_export_json=None):
        super().__init__(master, bg=THEME["bg_dark"], width=360)
        self.on_export_csv = on_export_csv
        self.on_export_json = on_export_json
        self.pack_propagate(False)
        self._build()

    def _build(self):
        tk.Label(
            self, text="RGB 点云", bg=THEME["bg_dark"], fg=THEME["text_primary"],
            font=FONT_SECTION, anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(10, 4))

        self.mode_label = tk.Label(
            self, text="未生成", bg=THEME["bg_dark"], fg=THEME["text_muted"],
            font=(THEME["font_family"], 10),
        )
        self.mode_label.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.fig = Figure(figsize=(3.5, 3.5), dpi=80, facecolor=THEME["bg_input"])
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(THEME["bg_input"])
        self._style_axes()
        self.fig.tight_layout(pad=1.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        self.info_label = tk.Label(
            self, text="", bg=THEME["bg_dark"], fg=THEME["text_secondary"],
            font=FONT_MONO,
        )
        self.info_label.pack(fill=tk.X, padx=10, pady=(0, 4))

        btn_frame = tk.Frame(self, bg=THEME["bg_dark"])
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        make_button(
            btn_frame, text="导出 CSV", style="default",
            command=lambda: self.on_export_csv and self.on_export_csv(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        make_button(
            btn_frame, text="导出 JSON", style="default",
            command=lambda: self.on_export_json and self.on_export_json(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _style_axes(self):
        self.ax.set_xlabel("R", color="#ff6666", fontsize=8, labelpad=2)
        self.ax.set_ylabel("G", color="#66ff66", fontsize=8, labelpad=2)
        self.ax.set_zlabel("B", color="#6688ff", fontsize=8, labelpad=2)
        self.ax.tick_params(colors=THEME["text_muted"], labelsize=6)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor(THEME["border"])
        self.ax.yaxis.pane.set_edgecolor(THEME["border"])
        self.ax.zaxis.pane.set_edgecolor(THEME["border"])
        self.ax.grid(True, color=THEME["border"], linewidth=0.3)

    def display_point_cloud(self, point_cloud: dict, title: str = ""):
        self.ax.clear()
        self._style_axes()

        if not point_cloud or point_cloud["count"] == 0:
            self.mode_label.config(text="无点云数据", fg=THEME["text_muted"])
            self.info_label.config(text="")
            self.canvas.draw()
            return

        points = np.array(point_cloud["points"])
        colors = np.array(point_cloud["colors"])

        r, g, b = points[:, 0], points[:, 1], points[:, 2]

        sample_size = min(len(points), PC_DISPLAY_SAMPLE_MAX)
        if len(points) > sample_size:
            idx = np.random.choice(len(points), sample_size, replace=False)
            r, g, b = r[idx], g[idx], b[idx]
            colors = colors[idx]

        self.ax.scatter(r, g, b, c=colors, s=2, alpha=0.6, depthshade=True)

        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
        self.ax.set_zlim(0, 255)

        self.mode_label.config(text=title or "RGB 点云", fg=THEME["text_primary"])
        self.info_label.config(text=f"采样点: {point_cloud['count']}")
        self.canvas.draw()

    def clear(self):
        self.ax.clear()
        self._style_axes()
        self.mode_label.config(text="未生成", fg=THEME["text_muted"])
        self.info_label.config(text="")
        self.canvas.draw()
