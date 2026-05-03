import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from models.annotation import Annotation, Project
from models.config import get, set_value, save_config, load_config, reset, DEFAULTS
from models.color_extractor import (
    extract_bbox_pixels, _rgb_to_hsv_vectorized, _find_dominant_colors,
    compute_color_stats, anomaly_detect_annotations,
)
from models.logger import OperationLogger
from models.exporter import export_coco, export_voc, export_yolo
from models.point_cloud import (
    sample_image_pixels, sample_bbox_pixels, pixels_to_point_cloud,
    generate_image_point_cloud, generate_bbox_point_cloud,
    export_point_cloud_csv, export_point_cloud_json,
)
from utils import (
    safe_close_pil, snapshot_annotations, restore_annotations,
    SUPPORTED_IMAGE_EXTS, THEME, UNDO_STACK_MAX, CLICK_CYCLE_TIMEOUT,
)


class TestAnnotation(unittest.TestCase):
    def test_create_with_id(self):
        ann = Annotation(bbox=(0.1, 0.2, 0.3, 0.4), label="cat", ann_id="abc123")
        self.assertEqual(ann.id, "abc123")
        self.assertEqual(ann.bbox, (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(ann.label, "cat")

    def test_create_auto_id(self):
        ann = Annotation(bbox=(0, 0, 1, 1), label="dog")
        self.assertEqual(len(ann.id), 12)

    def test_to_dict_roundtrip(self):
        ann = Annotation(
            bbox=(0.1, 0.2, 0.5, 0.6), label="cat",
            attributes={"occluded": True},
            color_stats={"mean_rgb": [128, 64, 32]},
            color_verified=True, ann_id="test123",
        )
        d = ann.to_dict()
        self.assertEqual(d["id"], "test123")
        self.assertEqual(d["bbox"], [0.1, 0.2, 0.5, 0.6])
        self.assertEqual(d["label"], "cat")
        self.assertTrue(d["color_verified"])
        self.assertIn("color_stats", d)

        ann2 = Annotation.from_dict(d)
        self.assertEqual(ann2.id, ann.id)
        self.assertEqual(list(ann2.bbox), list(ann.bbox))
        self.assertEqual(ann2.label, ann.label)
        self.assertEqual(ann2.attributes, ann.attributes)
        self.assertTrue(ann2.color_verified)

    def test_area(self):
        ann = Annotation(bbox=(0.1, 0.2, 0.5, 0.6), label="x")
        self.assertAlmostEqual(ann.area(), 0.4 * 0.4)


class TestProject(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_labels(self):
        p = Project(self.tmpdir)
        p.add_label("cat")
        p.add_label("dog")
        p.add_label("cat")
        self.assertEqual(p.labels, ["cat", "dog"])
        p.remove_label("cat")
        self.assertEqual(p.labels, ["dog"])

    def test_annotations_crud(self):
        p = Project(self.tmpdir)
        ann = Annotation(bbox=(0, 0, 1, 1), label="x", ann_id="a1")
        p.add_annotation("img.jpg", ann)
        self.assertEqual(len(p.get_annotations("img.jpg")), 1)
        p.add_annotation("img.jpg", Annotation(bbox=(0.1, 0.1, 0.5, 0.5), label="y", ann_id="a2"))
        self.assertEqual(len(p.get_annotations("img.jpg")), 2)
        p.remove_annotation("img.jpg", 0)
        self.assertEqual(len(p.get_annotations("img.jpg")), 1)
        self.assertEqual(p.get_annotations("img.jpg")[0].id, "a2")

    def test_save_load_roundtrip(self):
        p = Project(self.tmpdir)
        p.add_label("cat")
        p.add_label("dog")
        ann = Annotation(bbox=(0.1, 0.2, 0.3, 0.4), label="cat", ann_id="r1")
        p.add_annotation("test.jpg", ann)
        p.save()

        p2 = Project(self.tmpdir)
        p2.load()
        self.assertEqual(p2.labels, ["cat", "dog"])
        anns = p2.get_annotations("test.jpg")
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0].id, "r1")
        self.assertEqual(anns[0].label, "cat")

    def test_export_json(self):
        p = Project(self.tmpdir)
        p.add_annotation("a.jpg", Annotation(bbox=(0, 0, 1, 1), label="x", ann_id="e1"))
        out = os.path.join(self.tmpdir, "export.json")
        p.export_json(out)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["image"], "a.jpg")
        self.assertEqual(len(data[0]["annotations"]), 1)

    def test_export_csv(self):
        p = Project(self.tmpdir)
        p.add_annotation("a.jpg", Annotation(bbox=(0.1, 0.2, 0.3, 0.4), label="cat", ann_id="c1"))
        out = os.path.join(self.tmpdir, "export.csv")
        p.export_csv(out)
        with open(out, encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("a.jpg", lines[1])
        self.assertIn("cat", lines[1])

    def test_get_image_files_uses_supported_exts(self):
        self.assertEqual(SUPPORTED_IMAGE_EXTS, {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})


class TestConfig(unittest.TestCase):
    def setUp(self):
        reset()

    def test_get_default(self):
        self.assertEqual(get("z_score_threshold"), 2.5)
        self.assertEqual(get("undo_stack_size"), 50)
        self.assertIsNone(get("nonexistent"))

    def test_get_with_default(self):
        self.assertEqual(get("nonexistent", 42), 42)

    def test_set_value(self):
        set_value("z_score_threshold", 3.0)
        self.assertEqual(get("z_score_threshold"), 3.0)

    def test_save_load(self):
        tmpdir = tempfile.mkdtemp()
        try:
            set_value("z_score_threshold", 1.5)
            path = os.path.join(tmpdir, "config.json")
            save_config(path)
            reset()
            self.assertEqual(get("z_score_threshold"), 2.5)
            load_config(path)
            self.assertEqual(get("z_score_threshold"), 1.5)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_reset(self):
        set_value("z_score_threshold", 9.9)
        reset()
        self.assertEqual(get("z_score_threshold"), 2.5)

    def test_defaults_complete(self):
        expected_keys = {
            "z_score_threshold", "iqr_multiplier", "mahalanobis_threshold",
            "min_bbox_pixels", "max_bbox_ratio", "color_coverage_min",
            "brightness_variance_max", "cross_image_cv_max",
            "undo_stack_size", "point_cloud_sample_size", "point_cloud_bbox_sample_size",
        }
        self.assertEqual(set(DEFAULTS.keys()), expected_keys)


class TestUtils(unittest.TestCase):
    def test_safe_close_pil_none(self):
        safe_close_pil(None)

    def test_safe_close_pil_valid(self):
        img = Image.new("RGB", (10, 10))
        safe_close_pil(img)

    def test_snapshot_restore_roundtrip(self):
        anns = [
            Annotation(bbox=(0.1, 0.2, 0.3, 0.4), label="cat", ann_id="s1"),
            Annotation(bbox=(0.5, 0.6, 0.7, 0.8), label="dog", ann_id="s2"),
        ]
        snap = snapshot_annotations(anns)
        self.assertIsInstance(snap, list)
        self.assertEqual(len(snap), 2)
        self.assertIsInstance(snap[0], dict)

        restored = restore_annotations(snap)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].id, "s1")
        self.assertEqual(restored[0].label, "cat")
        self.assertEqual(restored[1].id, "s2")
        self.assertEqual(list(restored[1].bbox), [0.5, 0.6, 0.7, 0.8])

    def test_theme_keys(self):
        required = {
            "bg_dark", "bg_panel", "bg_input", "bg_control", "bg_hover",
            "accent", "accent_hover", "danger", "danger_hover",
            "text_primary", "text_secondary", "text_muted", "border",
        }
        self.assertTrue(required.issubset(set(THEME.keys())))

    def test_constants(self):
        self.assertEqual(UNDO_STACK_MAX, 50)
        self.assertEqual(CLICK_CYCLE_TIMEOUT, 0.5)
        self.assertEqual(SUPPORTED_IMAGE_EXTS, {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"})


class TestColorExtractor(unittest.TestCase):
    def _make_test_image(self, w=50, h=50, color=(128, 64, 32)):
        return Image.new("RGB", (w, h), color)

    def test_extract_bbox_pixels(self):
        img = self._make_test_image(100, 100, (200, 100, 50))
        pixels = extract_bbox_pixels(img, (0.1, 0.1, 0.5, 0.5))
        self.assertEqual(pixels.shape[1], 3)
        self.assertGreater(len(pixels), 0)
        np.testing.assert_array_equal(pixels[0], [200, 100, 50])

    def test_extract_bbox_empty(self):
        img = self._make_test_image(100, 100)
        pixels = extract_bbox_pixels(img, (0.5, 0.5, 0.1, 0.1))
        self.assertEqual(len(pixels), 0)

    def test_hsv_vectorized_pure_red(self):
        pixels = np.array([[255, 0, 0]], dtype=np.uint8)
        h, s, v = _rgb_to_hsv_vectorized(pixels)
        self.assertAlmostEqual(h[0], 0.0, places=0)
        self.assertAlmostEqual(s[0], 1.0, places=1)
        self.assertAlmostEqual(v[0], 1.0, places=1)

    def test_hsv_vectorized_pure_green(self):
        pixels = np.array([[0, 255, 0]], dtype=np.uint8)
        h, s, v = _rgb_to_hsv_vectorized(pixels)
        self.assertAlmostEqual(h[0], 120.0, delta=1.0)
        self.assertAlmostEqual(s[0], 1.0, places=1)

    def test_hsv_vectorized_pure_blue(self):
        pixels = np.array([[0, 0, 255]], dtype=np.uint8)
        h, s, v = _rgb_to_hsv_vectorized(pixels)
        self.assertAlmostEqual(h[0], 240.0, delta=1.0)
        self.assertAlmostEqual(s[0], 1.0, places=1)

    def test_hsv_vectorized_gray(self):
        pixels = np.array([[128, 128, 128]], dtype=np.uint8)
        h, s, v = _rgb_to_hsv_vectorized(pixels)
        self.assertAlmostEqual(s[0], 0.0, places=1)

    def test_hsv_vectorized_white(self):
        pixels = np.array([[255, 255, 255]], dtype=np.uint8)
        h, s, v = _rgb_to_hsv_vectorized(pixels)
        self.assertAlmostEqual(s[0], 0.0, places=1)
        self.assertAlmostEqual(v[0], 1.0, places=1)

    def test_hsv_vectorized_batch(self):
        pixels = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
        ], dtype=np.uint8)
        h, s, v = _rgb_to_hsv_vectorized(pixels)
        self.assertEqual(len(h), 4)
        self.assertEqual(len(s), 4)
        self.assertEqual(len(v), 4)
        self.assertAlmostEqual(h[0], 0.0, delta=1.0)
        self.assertAlmostEqual(h[1], 120.0, delta=1.0)
        self.assertAlmostEqual(h[2], 240.0, delta=1.0)

    def test_find_dominant_colors(self):
        pixels = np.tile([255, 0, 0], (100, 1)).astype(np.uint8)
        result = _find_dominant_colors(pixels, n_colors=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rgb"], [255, 0, 0])
        self.assertEqual(result[0]["hex"], "#ff0000")
        self.assertAlmostEqual(result[0]["ratio"], 1.0, places=1)

    def test_find_dominant_colors_mixed(self):
        red = np.tile([255, 0, 0], (80, 1)).astype(np.uint8)
        blue = np.tile([0, 0, 255], (20, 1)).astype(np.uint8)
        pixels = np.vstack([red, blue])
        result = _find_dominant_colors(pixels, n_colors=2)
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0]["ratio"], result[1]["ratio"])

    def test_compute_color_stats(self):
        img = self._make_test_image(40, 40, (100, 150, 200))
        stats = compute_color_stats(img, (0.0, 0.0, 1.0, 1.0))
        self.assertIn("mean_rgb", stats)
        self.assertIn("std_rgb", stats)
        self.assertIn("brightness_mean", stats)
        self.assertIn("saturation_mean", stats)
        self.assertIn("histogram", stats)
        self.assertIn("dominant_colors", stats)
        self.assertIn("hsv", stats)
        self.assertIn("pixel_count", stats)
        self.assertEqual(stats["pixel_count"], 40 * 40)
        self.assertAlmostEqual(stats["mean_rgb"][0], 100.0, delta=1.0)
        self.assertAlmostEqual(stats["mean_rgb"][1], 150.0, delta=1.0)
        self.assertAlmostEqual(stats["mean_rgb"][2], 200.0, delta=1.0)

    def test_compute_color_stats_empty(self):
        img = self._make_test_image(10, 10)
        stats = compute_color_stats(img, (0.5, 0.5, 0.1, 0.1))
        self.assertEqual(stats, {"error": "empty_region"})

    def test_hsv_stats_in_color_stats(self):
        img = self._make_test_image(30, 30, (200, 50, 50))
        stats = compute_color_stats(img, (0.0, 0.0, 1.0, 1.0))
        hsv = stats["hsv"]
        self.assertIn("hue_mean", hsv)
        self.assertIn("saturation_hsv_mean", hsv)
        self.assertIn("value_mean", hsv)
        self.assertAlmostEqual(hsv["hue_mean"], 0.0, delta=5.0)
        self.assertGreater(hsv["saturation_hsv_mean"], 0.5)

    def test_anomaly_detect_insufficient(self):
        ann = Annotation(bbox=(0, 0, 1, 1), label="x")
        ann.color_stats = {
            "mean_rgb": [128, 64, 32],
            "std_rgb": [10, 10, 10],
            "brightness_mean": 0.4,
            "saturation_mean": 0.3,
            "hsv": {"hue_mean": 30, "hue_std": 5, "saturation_hsv_mean": 0.5, "value_mean": 0.5},
        }
        result = anomaly_detect_annotations([ann])
        self.assertEqual(result["method"], "insufficient_data")
        self.assertEqual(result["anomalies"], [])

    def test_anomaly_detect_normal(self):
        anns = []
        for i in range(10):
            a = Annotation(bbox=(0, 0, 1, 1), label="x", ann_id=f"a{i}")
            a.color_stats = {
                "mean_rgb": [128, 64, 32],
                "std_rgb": [10, 10, 10],
                "brightness_mean": 0.4,
                "saturation_mean": 0.3,
                "hsv": {"hue_mean": 30, "hue_std": 5, "saturation_hsv_mean": 0.5, "value_mean": 0.5},
            }
            anns.append(a)
        result = anomaly_detect_annotations(anns)
        self.assertEqual(result["method"], "z_score + iqr + mahalanobis")
        self.assertEqual(result["total"], 10)
        self.assertEqual(result["anomaly_count"], 0)
        self.assertEqual(result["normal_count"], 10)


class TestOperationLogger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = OperationLogger(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_start_end_session(self):
        self.logger.start_session()
        self.assertIsNotNone(self.logger._session_start)
        self.assertEqual(len(self.logger._events), 1)
        self.assertEqual(self.logger._events[0]["type"], "SESSION_START")
        self.logger.end_session()
        self.assertEqual(self.logger._events[-1]["type"], "SESSION_END")

    def test_log_open_image(self):
        self.logger.start_session()
        self.logger.log_open_image("test.jpg")
        self.assertEqual(self.logger._current_image, "test.jpg")
        self.assertEqual(self.logger._events[-1]["type"], "IMAGE_OPEN")

    def test_log_open_image_switch(self):
        self.logger.start_session()
        self.logger.log_open_image("a.jpg")
        self.logger.log_open_image("b.jpg")
        leave_events = [e for e in self.logger._events if e["type"] == "IMAGE_LEAVE"]
        self.assertEqual(len(leave_events), 1)
        self.assertEqual(leave_events[0]["data"]["image"], "a.jpg")

    def test_log_create_annotation(self):
        self.logger.start_session()
        self.logger.log_create_annotation("img.jpg", "cat", (0.1, 0.2, 0.3, 0.4), 0, ann_id="a1")
        e = self.logger._events[-1]
        self.assertEqual(e["type"], "ANNOTATION_CREATE")
        self.assertEqual(e["data"]["label"], "cat")
        self.assertEqual(e["data"]["ann_id"], "a1")

    def test_log_move_annotation(self):
        self.logger.start_session()
        self.logger.log_move_annotation("img.jpg", 0, (0.1, 0.2, 0.3, 0.4), (0.2, 0.3, 0.4, 0.5))
        e = self.logger._events[-1]
        self.assertEqual(e["type"], "ANNOTATION_MOVE")
        self.assertEqual(e["data"]["old_bbox"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(e["data"]["new_bbox"], [0.2, 0.3, 0.4, 0.5])

    def test_log_verify(self):
        self.logger.start_session()
        self.logger.log_verify("img.jpg", 5, "all_pass")
        e = self.logger._events[-1]
        self.assertEqual(e["type"], "VERIFY")
        self.assertEqual(e["data"]["annotation_count"], 5)

    def test_save_creates_file(self):
        self.logger.start_session()
        self.logger.log_create_annotation("img.jpg", "cat", (0.1, 0.2, 0.3, 0.4), 0)
        path = self.logger.save()
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("events", data)
        self.assertIn("statistics", data)
        self.assertEqual(data["statistics"]["annotations_created"], 1)

    def test_compute_stats_label_distribution(self):
        self.logger.start_session()
        self.logger.log_create_annotation("a.jpg", "cat", (0, 0, 1, 1), 0)
        self.logger.log_create_annotation("a.jpg", "cat", (0, 0, 1, 1), 1)
        self.logger.log_create_annotation("a.jpg", "dog", (0, 0, 1, 1), 2)
        self.logger.log_export("coco", "/tmp/out.json", 3)
        self.logger.log_undo("delete")
        stats = self.logger._compute_stats()
        self.assertEqual(stats["annotations_created"], 3)
        self.assertEqual(stats["label_distribution"], {"cat": 2, "dog": 1})
        self.assertEqual(stats["exports"], 1)

    def test_log_undo_redo(self):
        self.logger.start_session()
        self.logger.log_undo("delete_ann")
        self.logger.log_redo("delete_ann")
        types = [e["type"] for e in self.logger._events[-2:]]
        self.assertEqual(types, ["UNDO", "REDO"])


class TestExporter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Project(self.tmpdir)
        self.project.add_label("cat")
        self.project.add_label("dog")
        self.project.add_annotation("a.jpg", Annotation(bbox=(0.1, 0.2, 0.3, 0.4), label="cat", ann_id="e1"))
        self.project.add_annotation("a.jpg", Annotation(bbox=(0.5, 0.6, 0.8, 0.9), label="dog", ann_id="e2"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_coco_format(self):
        out = os.path.join(self.tmpdir, "coco.json")
        n_img, n_ann, n_cat = export_coco(self.project, out)
        self.assertTrue(os.path.exists(out))
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("images", data)
        self.assertIn("annotations", data)
        self.assertIn("categories", data)
        self.assertEqual(n_ann, 2)
        self.assertEqual(n_cat, 2)

    def test_export_coco_categories(self):
        out = os.path.join(self.tmpdir, "coco.json")
        export_coco(self.project, out)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        cat_names = {c["name"] for c in data["categories"]}
        self.assertEqual(cat_names, {"cat", "dog"})

    def test_export_voc_creates_xml(self):
        out_dir = os.path.join(self.tmpdir, "voc")
        count = export_voc(self.project, out_dir)
        self.assertEqual(count, 1)
        xml_path = os.path.join(out_dir, "a.xml")
        self.assertTrue(os.path.exists(xml_path))
        with open(xml_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("cat", content)
        self.assertIn("dog", content)

    def test_export_yolo_format(self):
        out_dir = os.path.join(self.tmpdir, "yolo")
        n_files, n_anns = export_yolo(self.project, out_dir)
        self.assertEqual(n_files, 1)
        self.assertEqual(n_anns, 2)
        classes_path = os.path.join(out_dir, "classes.txt")
        self.assertTrue(os.path.exists(classes_path))
        with open(classes_path, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        self.assertEqual(lines, ["cat", "dog"])
        txt_path = os.path.join(out_dir, "a.txt")
        self.assertTrue(os.path.exists(txt_path))
        with open(txt_path, encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        parts = lines[0].strip().split()
        self.assertEqual(len(parts), 5)
        self.assertEqual(int(parts[0]), 0)

    def test_export_yolo_extra_labels(self):
        self.project.add_annotation("b.jpg", Annotation(bbox=(0, 0, 1, 1), label="fish", ann_id="e3"))
        out_dir = os.path.join(self.tmpdir, "yolo2")
        export_yolo(self.project, out_dir)
        with open(os.path.join(out_dir, "classes.txt"), encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        self.assertIn("fish", lines)


class TestPointCloud(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (50, 50), (100, 150, 200))

    def test_sample_image_pixels_full(self):
        pixels = sample_image_pixels(self.img, n_samples=10000)
        self.assertEqual(pixels.shape, (2500, 3))
        np.testing.assert_array_equal(pixels[0], [100, 150, 200])

    def test_sample_image_pixels_subsample(self):
        pixels = sample_image_pixels(self.img, n_samples=100)
        self.assertEqual(pixels.shape[0], 100)
        self.assertEqual(pixels.shape[1], 3)

    def test_sample_image_pixels_empty(self):
        img = Image.new("RGB", (0, 0))
        pixels = sample_image_pixels(img, n_samples=100)
        self.assertEqual(len(pixels), 0)

    def test_pixels_to_point_cloud_basic(self):
        pixels = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        pc = pixels_to_point_cloud(pixels)
        self.assertEqual(pc["count"], 2)
        self.assertEqual(len(pc["points"]), 2)
        self.assertEqual(len(pc["colors"]), 2)
        self.assertEqual(pc["points"][0], [255.0, 0.0, 0.0])
        self.assertAlmostEqual(pc["colors"][0][0], 1.0, places=2)

    def test_pixels_to_point_cloud_empty(self):
        pixels = np.array([], dtype=np.uint8).reshape(0, 3)
        pc = pixels_to_point_cloud(pixels)
        self.assertEqual(pc["count"], 0)
        self.assertEqual(pc["points"], [])

    def test_generate_image_point_cloud(self):
        pc = generate_image_point_cloud(self.img, n_samples=100)
        self.assertEqual(pc["count"], 100)
        self.assertGreater(len(pc["points"]), 0)

    def test_export_point_cloud_csv(self):
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "pc.csv")
            pc = pixels_to_point_cloud(np.array([[255, 0, 0]], dtype=np.uint8))
            export_point_cloud_csv(path, pc)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("r", lines[0])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_export_point_cloud_json(self):
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "pc.json")
            pc = pixels_to_point_cloud(np.array([[255, 0, 0]], dtype=np.uint8))
            export_point_cloud_json(path, pc)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["count"], 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
