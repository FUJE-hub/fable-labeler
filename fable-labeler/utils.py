import os
import json
import tkinter as tk
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

THEME = {
    "bg_dark": "#1e1e1e",
    "bg_panel": "#252526",
    "bg_input": "#2d2d2d",
    "bg_control": "#3c3c3c",
    "bg_hover": "#505050",
    "accent": "#0078d4",
    "accent_hover": "#1a8cff",
    "danger": "#c42b1c",
    "danger_hover": "#e04030",
    "warning": "#ff9800",
    "success": "#4ec9b0",
    "pass_color": "#4caf50",
    "warn_color": "#ff9800",
    "fail_color": "#f44336",
    "text_primary": "#e0e0e0",
    "text_secondary": "#a0a0a0",
    "text_muted": "#707070",
    "border": "#3a3a3a",
    "font_family": "Segoe UI",
    "font_mono": "Consolas",
    "radius": 4,
}

UNDO_STACK_MAX = 50
CLICK_CYCLE_TIMEOUT = 0.5
DEFAULT_WINDOW_SIZE = "1440x820"

FONT_LABEL = (THEME["font_family"], 10)
FONT_LABEL_BOLD = (THEME["font_family"], 10, "bold")
FONT_SECTION = (THEME["font_family"], 11, "bold")
FONT_BUTTON = (THEME["font_family"], 9)
FONT_MONO = (THEME["font_mono"], 9)
FONT_STATUS = (THEME["font_family"], 9)
FONT_CANVAS_LABEL = (THEME["font_family"], 9, "bold")


def get_image_size(image_path):
    with Image.open(image_path) as img:
        img.load()
        return img.size


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_close_pil(img):
    if img is None:
        return
    try:
        img.close()
    except Exception:
        pass


def center_window(root, width, height):
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - width) // 2
    y = (sh - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")


def make_button(parent, text, command=None, style="default", **kwargs):
    styles = {
        "primary": {"bg": THEME["accent"], "fg": "#ffffff", "activebackground": THEME["accent_hover"]},
        "danger": {"bg": THEME["danger"], "fg": "#ffffff", "activebackground": THEME["danger_hover"]},
        "default": {"bg": THEME["bg_control"], "fg": THEME["text_primary"]},
        "ghost": {"bg": THEME["bg_panel"], "fg": THEME["text_secondary"]},
    }
    s = styles.get(style, styles["default"])
    defaults = {
        "font": FONT_BUTTON,
        "borderwidth": 0,
        "cursor": "hand2",
        "relief": "flat",
        "padx": 6,
        "pady": 3,
    }
    defaults.update(s)
    defaults.update(kwargs)
    btn = tk.Button(parent, text=text, command=command, **defaults)
    btn.pack_propagate(False)
    return btn


def make_label(parent, text, size="normal", **kwargs):
    sizes = {
        "section": {"font": FONT_SECTION, "fg": THEME["text_primary"]},
        "normal": {"font": FONT_LABEL, "fg": THEME["text_primary"]},
        "muted": {"font": FONT_STATUS, "fg": THEME["text_muted"]},
        "mono": {"font": FONT_MONO, "fg": THEME["text_secondary"]},
    }
    s = sizes.get(size, sizes["normal"])
    defaults = {"bg": THEME["bg_dark"], "anchor": "w"}
    defaults.update(s)
    defaults.update(kwargs)
    return tk.Label(parent, text=text, **defaults)


def snapshot_annotations(annotations):
    return [a.to_dict() for a in annotations]


def restore_annotations(snapshot):
    from models.annotation import Annotation
    return [Annotation.from_dict(d) for d in snapshot]
