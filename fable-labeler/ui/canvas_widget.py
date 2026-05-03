import time
import gc
import tkinter as tk
from PIL import Image, ImageTk

from utils import THEME, FONT_CANVAS_LABEL, safe_close_pil, CLICK_CYCLE_TIMEOUT

HANDLE_SIZE = 6
MIN_DRAG_PX = 5
MIN_BBOX_RATIO = 0.005
ZOOM_MIN = 0.1
ZOOM_MAX = 20.0
ZOOM_STEP = 1.1

COLOR_SELECTED = "#00ff88"
COLOR_NORMAL = "#ff4444"
COLOR_DRAW_GUIDE = "#ffff00"
BBOX_WIDTH_SELECTED = 3
BBOX_WIDTH_NORMAL = 2
DASH_GUIDE = (4, 4)


class CanvasWidget(tk.Frame):
    def __init__(self, master, on_box_created=None, on_box_selected=None,
                 on_box_moved=None, on_box_resized=None, on_drag_start=None):
        super().__init__(master, bg=THEME["bg_dark"])
        self.on_box_created = on_box_created
        self.on_box_selected = on_box_selected
        self.on_box_moved = on_box_moved
        self.on_box_resized = on_box_resized
        self.on_drag_start = on_drag_start

        self.canvas = tk.Canvas(self, bg=THEME["bg_input"], highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.tk_image = None
        self.image_id = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self._cached_scale = 0
        self._cached_photo = None

        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None

        self.annotations = []
        self.selected_index = -1
        self._ann_canvas_ids = {}

        self._drag_mode = None
        self._drag_handle = None
        self._drag_start_bbox = None
        self._drag_start_cx = 0
        self._drag_start_cy = 0
        self._last_click_idx = -1
        self._last_click_time = 0

        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_end)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def load_image(self, image_path):
        self._release_image_resources()
        self.image = Image.open(image_path)
        self.image.load()
        self._reset_draw_state()
        self.fit_image()
        self.redraw()
        return self.image.copy()

    def _release_image_resources(self):
        if self._cached_photo is not None:
            self._cached_photo = None
        if self.image is not None:
            safe_close_pil(self.image)
            self.image = None
        self.tk_image = None
        self.image_id = None
        self._cached_scale = 0
        gc.collect()

    def _reset_draw_state(self):
        self.annotations = []
        self.selected_index = -1
        self._drag_mode = None
        self._drag_handle = None
        self._drag_start_bbox = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._ann_canvas_ids.clear()

    def fit_image(self):
        if self.image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 600
        iw, ih = self.image.size
        base_scale = min(cw / iw, ch / ih)
        self.scale = base_scale * self._zoom
        self.offset_x = (cw - iw * self.scale) / 2 + self._pan_x
        self.offset_y = (ch - ih * self.scale) / 2 + self._pan_y

    def _render_image(self):
        new_scale = round(self.scale * 1000)
        if new_scale != self._cached_scale or self._cached_photo is None:
            w = max(1, int(self.image.width * self.scale))
            h = max(1, int(self.image.height * self.scale))
            resized = self.image.resize((w, h), Image.NEAREST)
            self._cached_photo = ImageTk.PhotoImage(resized)
            self._cached_scale = new_scale
        self.tk_image = self._cached_photo
        self.image_id = self.canvas.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image
        )

    def redraw(self):
        self.canvas.delete("all")
        self._ann_canvas_ids.clear()
        if self.image is None:
            return
        self.fit_image()
        self._render_image()
        for i, ann in enumerate(self.annotations):
            self._draw_annotation(i, ann)

    def _draw_annotation(self, index, ann):
        sx1, sy1, sx2, sy2 = self._bbox_to_canvas(ann.bbox)
        is_sel = (index == self.selected_index)
        color = COLOR_SELECTED if is_sel else COLOR_NORMAL
        width = BBOX_WIDTH_SELECTED if is_sel else BBOX_WIDTH_NORMAL

        rect_id = self.canvas.create_rectangle(
            sx1, sy1, sx2, sy2, outline=color, width=width, tags=f"bbox_{index}"
        )
        label_text = f"[{index+1}] {ann.label}" if ann.label else f"[{index+1}] unlabeled"
        text_id = self.canvas.create_text(
            sx1, sy1 - 2, anchor=tk.SW, text=label_text,
            fill=color, font=FONT_CANVAS_LABEL, tags=f"bbox_label_{index}",
        )

        handle_ids = []
        if is_sel:
            hs = HANDLE_SIZE
            handles = self._compute_handle_positions(sx1, sy1, sx2, sy2)
            for _, hx, hy in handles:
                hid = self.canvas.create_rectangle(
                    hx - hs, hy - hs, hx + hs, hy + hs,
                    fill=COLOR_SELECTED, outline="#ffffff", width=1,
                )
                handle_ids.append(hid)

        self._ann_canvas_ids[index] = {
            "rect": rect_id, "text": text_id, "handles": handle_ids,
        }

    def _bbox_to_canvas(self, bbox):
        x1, y1, x2, y2 = bbox
        iw, ih = self.image.width, self.image.height
        s = self.scale
        return (
            self.offset_x + x1 * iw * s,
            self.offset_y + y1 * ih * s,
            self.offset_x + x2 * iw * s,
            self.offset_y + y2 * ih * s,
        )

    def _canvas_to_image(self, cx, cy):
        if self.image is None:
            return 0, 0
        ix = (cx - self.offset_x) / (self.image.width * self.scale)
        iy = (cy - self.offset_y) / (self.image.height * self.scale)
        return ix, iy

    @staticmethod
    def _compute_handle_positions(sx1, sy1, sx2, sy2):
        mx = (sx1 + sx2) / 2
        my = (sy1 + sy2) / 2
        return [
            ("nw", sx1, sy1), ("n", mx, sy1), ("ne", sx2, sy1),
            ("w", sx1, my), ("e", sx2, my),
            ("sw", sx1, sy2), ("s", mx, sy2), ("se", sx2, sy2),
        ]

    def _hit_test_handle(self, cx, cy):
        if self.selected_index < 0 or self.selected_index >= len(self.annotations):
            return None
        ann = self.annotations[self.selected_index]
        sx1, sy1, sx2, sy2 = self._bbox_to_canvas(ann.bbox)
        hs = HANDLE_SIZE + 4
        for name, hx, hy in self._compute_handle_positions(sx1, sy1, sx2, sy2):
            if abs(cx - hx) <= hs and abs(cy - hy) <= hs:
                return name
        return None

    def _hit_test_annotations(self, cx, cy, exclude=-1):
        hits = []
        for i in range(len(self.annotations) - 1, -1, -1):
            if i == exclude:
                continue
            sx1, sy1, sx2, sy2 = self._bbox_to_canvas(self.annotations[i].bbox)
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                hits.append(i)
        return hits

    def _resolve_hit(self, hits):
        now = time.time()
        if (self._last_click_idx in hits and len(hits) > 1
                and now - self._last_click_time < CLICK_CYCLE_TIMEOUT):
            idx_in_hits = hits.index(self._last_click_idx)
            hit_idx = hits[(idx_in_hits + 1) % len(hits)]
        else:
            hit_idx = hits[0]
        self._last_click_idx = hit_idx
        self._last_click_time = now
        return hit_idx

    def on_press(self, event):
        if self.image is None:
            return

        handle = self._hit_test_handle(event.x, event.y)
        if handle is not None:
            self._start_resize(handle, event)
            return

        hits = self._hit_test_annotations(event.x, event.y)
        if hits:
            self._start_move(hits, event)
            return

        self._clear_selection()
        self._start_draw(event)

    def _start_resize(self, handle, event):
        self._drag_mode = "resize"
        self._drag_handle = handle
        self._drag_start_bbox = list(self.annotations[self.selected_index].bbox)
        self._drag_start_cx = event.x
        self._drag_start_cy = event.y
        if self.on_drag_start:
            self.on_drag_start(self.selected_index)

    def _start_move(self, hits, event):
        hit_idx = self._resolve_hit(hits)
        self.selected_index = hit_idx
        self._drag_mode = "move"
        self._drag_start_bbox = list(self.annotations[self.selected_index].bbox)
        self._drag_start_cx = event.x
        self._drag_start_cy = event.y
        if self.on_drag_start:
            self.on_drag_start(hit_idx)
        self.redraw()
        if self.on_box_selected:
            self.on_box_selected(hit_idx)

    def _clear_selection(self):
        self.selected_index = -1
        self._drag_mode = None
        self._last_click_idx = -1
        self.redraw()

    def _start_draw(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=COLOR_DRAW_GUIDE, width=2, dash=DASH_GUIDE, tags="temp_rect",
        )

    def _update_dragged_annotation(self):
        idx = self.selected_index
        if idx < 0 or idx not in self._ann_canvas_ids:
            self.redraw()
            return

        ann = self.annotations[idx]
        sx1, sy1, sx2, sy2 = self._bbox_to_canvas(ann.bbox)
        ids = self._ann_canvas_ids[idx]

        self.canvas.coords(ids["rect"], sx1, sy1, sx2, sy2)
        label_text = f"[{idx+1}] {ann.label}" if ann.label else f"[{idx+1}] unlabeled"
        self.canvas.coords(ids["text"], sx1, sy1 - 2)
        self.canvas.itemconfig(ids["text"], text=label_text)

        if ids["handles"]:
            hs = HANDLE_SIZE
            handles = self._compute_handle_positions(sx1, sy1, sx2, sy2)
            for j, (_, hx, hy) in enumerate(handles):
                if j < len(ids["handles"]):
                    self.canvas.coords(
                        ids["handles"][j], hx - hs, hy - hs, hx + hs, hy + hs,
                    )

    def on_drag(self, event):
        if self._drag_mode == "move" and self.selected_index >= 0:
            self._do_move(event)
            return
        if self._drag_mode == "resize" and self.selected_index >= 0:
            self._do_resize(event)
            return
        if self.drawing and self.current_rect:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def _do_move(self, event):
        dx = (event.x - self._drag_start_cx) / (self.image.width * self.scale)
        dy = (event.y - self._drag_start_cy) / (self.image.height * self.scale)
        x1, y1, x2, y2 = self._drag_start_bbox
        self.annotations[self.selected_index].bbox = (
            max(0.0, min(1.0, x1 + dx)),
            max(0.0, min(1.0, y1 + dy)),
            max(0.0, min(1.0, x2 + dx)),
            max(0.0, min(1.0, y2 + dy)),
        )
        self._update_dragged_annotation()

    def _do_resize(self, event):
        ix, iy = self._canvas_to_image(event.x, event.y)
        x1, y1, x2, y2 = self._drag_start_bbox
        handle = self._drag_handle
        if "w" in handle:
            x1 = max(0.0, min(1.0, ix))
        if "e" in handle:
            x2 = max(0.0, min(1.0, ix))
        if "n" in handle:
            y1 = max(0.0, min(1.0, iy))
        if "s" in handle:
            y2 = max(0.0, min(1.0, iy))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        self.annotations[self.selected_index].bbox = (x1, y1, x2, y2)
        self._update_dragged_annotation()

    def on_release(self, event):
        if self._drag_mode == "move" and self.selected_index >= 0:
            if self.on_box_moved:
                self.on_box_moved(self.selected_index, self.annotations[self.selected_index].bbox)
            self._drag_mode = None
            self.redraw()
            return
        if self._drag_mode == "resize" and self.selected_index >= 0:
            if self.on_box_resized:
                self.on_box_resized(self.selected_index, self.annotations[self.selected_index].bbox)
            self._drag_mode = None
            self._drag_handle = None
            self.redraw()
            return
        if not self.drawing:
            return
        self._finish_draw(event)

    def _finish_draw(self, event):
        self.drawing = False
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        dx = abs(event.x - self.start_x)
        dy = abs(event.y - self.start_y)
        if dx < MIN_DRAG_PX or dy < MIN_DRAG_PX or self.image is None:
            return

        ix1, iy1 = self._canvas_to_image(self.start_x, self.start_y)
        ix2, iy2 = self._canvas_to_image(event.x, event.y)

        x_min = max(0.0, min(ix1, ix2))
        y_min = max(0.0, min(iy1, iy2))
        x_max = min(1.0, max(ix1, ix2))
        y_max = min(1.0, max(iy1, iy2))

        if x_max - x_min < MIN_BBOX_RATIO or y_max - y_min < MIN_BBOX_RATIO:
            return

        if self.on_box_created:
            self.on_box_created((x_min, y_min, x_max, y_max))

    def on_mousewheel(self, event):
        if self.image is None:
            return
        factor = ZOOM_STEP if event.delta > 0 else (1 / ZOOM_STEP)
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * factor))
        self.redraw()

    def on_pan_start(self, event):
        self._panning = True
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def on_pan_move(self, event):
        if not self._panning:
            return
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        self._pan_x += dx
        self._pan_y += dy
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self.redraw()

    def on_pan_end(self, event):
        self._panning = False

    def reset_zoom(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.redraw()

    def set_annotations(self, annotations):
        self.annotations = annotations
        self.selected_index = -1
        self._drag_mode = None
        self._ann_canvas_ids.clear()
        self.redraw()

    def select_annotation(self, index):
        self.selected_index = index
        self._ann_canvas_ids.clear()
        self.redraw()

    def clear_selection(self):
        self.selected_index = -1
        self._drag_mode = None
        self._ann_canvas_ids.clear()
        self.redraw()
