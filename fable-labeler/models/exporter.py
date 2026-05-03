from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:
    from models.annotation import Project


def export_coco(project: Project, output_path: str) -> tuple[int, int, int]:
    images = []
    annotations = []
    categories = []

    label_to_id = {}
    for i, label in enumerate(project.labels, 1):
        label_to_id[label] = i
        categories.append({"id": i, "name": label, "supercategory": ""})

    ann_id = 1
    for img_id, (image_name, anns) in enumerate(project.image_annotations.items(), 1):
        image_path = os.path.join(project.project_dir, image_name)
        try:
            with Image.open(image_path) as img:
                img.load()
                w, h = img.size
        except Exception:
            w, h = 0, 0

        images.append({
            "id": img_id,
            "file_name": image_name,
            "width": w,
            "height": h,
        })

        for ann in anns:
            x1, y1, x2, y2 = ann.bbox
            abs_x1 = x1 * w
            abs_y1 = y1 * h
            abs_w = (x2 - x1) * w
            abs_h = (y2 - y1) * h
            cat_id = label_to_id.get(ann.label, 0)
            if cat_id == 0:
                label_to_id[ann.label] = len(label_to_id) + 1
                cat_id = label_to_id[ann.label]
                categories.append({"id": cat_id, "name": ann.label, "supercategory": ""})

            coco_ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(abs_x1, 2), round(abs_y1, 2), round(abs_w, 2), round(abs_h, 2)],
                "area": round(abs_w * abs_h, 2),
                "iscrowd": 0,
            }
            if ann.id:
                coco_ann["annotation_id"] = ann.id
            annotations.append(coco_ann)
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    return len(images), len(annotations), len(categories)


def export_voc(project: Project, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for image_name, anns in project.image_annotations.items():
        image_path = os.path.join(project.project_dir, image_name)
        try:
            with Image.open(image_path) as img:
                img.load()
                w, h = img.size
        except Exception:
            w, h = 0, 0

        root = ET.Element("annotation")

        fn = ET.SubElement(root, "filename")
        fn.text = image_name

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"

        for ann in anns:
            x1, y1, x2, y2 = ann.bbox
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = ann.label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(x1 * w))
            ET.SubElement(bndbox, "ymin").text = str(int(y1 * h))
            ET.SubElement(bndbox, "xmax").text = str(int(x2 * w))
            ET.SubElement(bndbox, "ymax").text = str(int(y2 * h))

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        base = os.path.splitext(image_name)[0]
        xml_path = os.path.join(output_dir, base + ".xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        count += 1

    return count


def export_yolo(project: Project, output_dir: str) -> tuple[int, int]:
    os.makedirs(output_dir, exist_ok=True)

    class_names_path = os.path.join(output_dir, "classes.txt")
    with open(class_names_path, "w", encoding="utf-8") as f:
        for label in project.labels:
            f.write(label + "\n")

    label_to_idx = {label: i for i, label in enumerate(project.labels)}

    file_count = 0
    ann_count = 0

    for image_name, anns in project.image_annotations.items():
        base = os.path.splitext(image_name)[0]
        txt_path = os.path.join(output_dir, base + ".txt")

        lines = []
        for ann in anns:
            cls_idx = label_to_idx.get(ann.label)
            if cls_idx is None:
                cls_idx = len(project.labels)
                label_to_idx[ann.label] = cls_idx

            x1, y1, x2, y2 = ann.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            ann_count += 1

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        file_count += 1

    extra_labels = [l for l in label_to_idx if l not in project.labels]
    if extra_labels:
        with open(class_names_path, "a", encoding="utf-8") as f:
            for l in extra_labels:
                f.write(l + "\n")

    return file_count, ann_count
