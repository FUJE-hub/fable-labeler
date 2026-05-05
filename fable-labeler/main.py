import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 高 DPI 适配
if sys.platform == "win32":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from ui.main_window import MainWindow


def main():
    root = tk.Tk()
    # 启用高分屏缩放
    root.tk.call("tk", "scaling", 1.0)
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
