import numpy as np
from PIL import Image
from collections import Counter
from models.config import get as cfg


BINS_PER_CHANNEL = 8
HSV_SAMPLE_MAX = 2000
DOMINANT_SAMPLE_MAX = 5000
HSV_BINS = 12
DOMINANT_S_THRESHOLDS = (0.2, 0.6)
DOMINANT_V_THRESHOLDS = (0.3, 0.7)


def extract_bbox_pixels(image: Image.Image, bbox: tuple) -> np.ndarray:
    iw, ih = image.size
    x1, y1, x2, y2 = bbox
    px1 = max(0, int(x1 * iw))
    py1 = max(0, int(y1 * ih))
    px2 = min(iw, int(x2 * iw))
    py2 = min(ih, int(y2 * ih))
    if px2 <= px1 or py2 <= py1:
        return np.array([], dtype=np.uint8).reshape(0, 3)
    cropped = image.crop((px1, py1, px2, py2))
    arr = np.array(cropped).reshape(-1, 3)
    return arr


def _compute_hsv_from_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple:
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 0
    idx = mask & (cmax == r)
    h[idx] = 60 * (((g[idx] - b[idx]) / delta[idx]) % 6)
    idx = mask & (cmax == g)
    h[idx] = 60 * ((b[idx] - r[idx]) / delta[idx] + 2)
    idx = mask & (cmax == b)
    h[idx] = 60 * ((r[idx] - g[idx]) / delta[idx] + 4)
    s = np.where(cmax > 0, delta / cmax, 0.0)
    return h, s, cmax


def _rgb_to_hsv_vectorized(pixels: np.ndarray) -> tuple:
    step = max(1, len(pixels) // HSV_SAMPLE_MAX)
    sample = pixels[::step].astype(np.float64) / 255.0
    r, g, b = sample[:, 0], sample[:, 1], sample[:, 2]
    return _compute_hsv_from_rgb(r, g, b)


def compute_color_stats(image: Image.Image, bbox: tuple) -> dict:
    pixels = extract_bbox_pixels(image, bbox)
    if len(pixels) == 0:
        return {"error": "empty_region"}

    r, g, b = pixels[:, 0].astype(float), pixels[:, 1].astype(float), pixels[:, 2].astype(float)

    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    std_r, std_g, std_b = float(np.std(r)), float(np.std(g)), float(np.std(b))

    norm = pixels.astype(float) / 255.0
    brightness = 0.299 * norm[:, 0] + 0.587 * norm[:, 1] + 0.114 * norm[:, 2]
    sat = np.max(norm, axis=1) - np.min(norm, axis=1)

    hist_r = np.histogram(r, bins=BINS_PER_CHANNEL, range=(0, 256))[0].tolist()
    hist_g = np.histogram(g, bins=BINS_PER_CHANNEL, range=(0, 256))[0].tolist()
    hist_b = np.histogram(b, bins=BINS_PER_CHANNEL, range=(0, 256))[0].tolist()

    dominant_colors = _find_dominant_colors(pixels, n_colors=5)

    hue, sat_hsv, val = _rgb_to_hsv_vectorized(pixels)
    hsv_stats = {
        "hue_mean": round(float(np.mean(hue)), 1),
        "hue_std": round(float(np.std(hue)), 1),
        "saturation_hsv_mean": round(float(np.mean(sat_hsv)), 4),
        "saturation_hsv_std": round(float(np.std(sat_hsv)), 4),
        "value_mean": round(float(np.mean(val)), 4),
        "value_std": round(float(np.std(val)), 4),
    }

    return {
        "mean_rgb": [round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)],
        "std_rgb": [round(std_r, 1), round(std_g, 1), round(std_b, 1)],
        "brightness_mean": round(float(np.mean(brightness)), 4),
        "brightness_std": round(float(np.std(brightness)), 4),
        "saturation_mean": round(float(np.mean(sat)), 4),
        "saturation_std": round(float(np.std(sat)), 4),
        "histogram": {"r": hist_r, "g": hist_g, "b": hist_b},
        "dominant_colors": dominant_colors,
        "hsv": hsv_stats,
        "pixel_count": int(len(pixels)),
    }


def _find_dominant_colors(pixels: np.ndarray, n_colors: int = 5) -> list:
    step = max(1, len(pixels) // DOMINANT_SAMPLE_MAX)
    sample = pixels[::step].astype(np.float64) / 255.0
    r, g, b = sample[:, 0], sample[:, 1], sample[:, 2]
    h, sat, val = _compute_hsv_from_rgb(r, g, b)

    h_bin = (h / 360 * HSV_BINS).astype(int).clip(0, HSV_BINS - 1)
    s_bin = np.where(sat < DOMINANT_S_THRESHOLDS[0], 0,
                     np.where(sat < DOMINANT_S_THRESHOLDS[1], 1, 2))
    v_bin = np.where(val < DOMINANT_V_THRESHOLDS[0], 0,
                     np.where(val < DOMINANT_V_THRESHOLDS[1], 1, 2))
    raw = pixels[::step]
    keys = list(zip(h_bin.tolist(), s_bin.tolist(), v_bin.tolist(),
                    raw[:, 0].tolist(), raw[:, 1].tolist(), raw[:, 2].tolist()))
    counter = Counter(keys)
    total = len(keys)
    results = []
    for key, count in counter.most_common(n_colors):
        results.append({
            "rgb": [key[3], key[4], key[5]],
            "hex": "#{:02x}{:02x}{:02x}".format(key[3], key[4], key[5]),
            "ratio": round(count / total, 4),
        })
    return results


def _z_score_detect(values: np.ndarray, threshold: float = None) -> dict:
    if threshold is None:
        threshold = cfg("z_score_threshold", 2.5)
    if len(values) < 3:
        return {"anomalies": [], "mean": 0, "std": 0, "z_scores": []}
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std = np.where(std == 0, 1e-6, std)
    z_scores = np.abs((values - mean) / std)
    anomaly_mask = np.any(z_scores > threshold, axis=1) if z_scores.ndim > 1 else z_scores > threshold
    return {
        "anomalies": np.where(anomaly_mask)[0].tolist(),
        "mean": mean.tolist() if hasattr(mean, 'tolist') else float(mean),
        "std": std.tolist() if hasattr(std, 'tolist') else float(std),
        "z_scores": z_scores.tolist() if hasattr(z_scores, 'tolist') else [float(z_scores)],
    }


def _iqr_detect(values: np.ndarray, multiplier: float = None) -> dict:
    if multiplier is None:
        multiplier = cfg("iqr_multiplier", 1.5)
    if len(values) < 4:
        return {"anomalies": [], "q1": 0, "q3": 0, "iqr": 0}
    q1 = np.percentile(values, 25, axis=0)
    q3 = np.percentile(values, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    if values.ndim > 1:
        anomaly_mask = np.any((values < lower) | (values > upper), axis=1)
    else:
        anomaly_mask = (values < lower) | (values > upper)
    return {
        "anomalies": np.where(anomaly_mask)[0].tolist(),
        "q1": q1.tolist() if hasattr(q1, 'tolist') else float(q1),
        "q3": q3.tolist() if hasattr(q3, 'tolist') else float(q3),
        "iqr": iqr.tolist() if hasattr(iqr, 'tolist') else float(iqr),
        "lower": lower.tolist() if hasattr(lower, 'tolist') else float(lower),
        "upper": upper.tolist() if hasattr(upper, 'tolist') else float(upper),
    }


def _mahalanobis_distance(point: np.ndarray, data: np.ndarray) -> float:
    if len(data) < 2:
        return 0.0
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    try:
        cov_inv = np.linalg.pinv(cov)
        diff = point - mean
        return float(np.sqrt(diff @ cov_inv @ diff))
    except np.linalg.LinAlgError:
        return 0.0


def _mahalanobis_detect(features: np.ndarray, threshold: float = None) -> dict:
    if threshold is None:
        threshold = cfg("mahalanobis_threshold", 5.0)
    if len(features) < 3:
        return {"anomalies": [], "distances": [], "mean_distance": 0.0, "threshold": threshold}
    distances = []
    for i in range(len(features)):
        others = np.delete(features, i, axis=0)
        d = _mahalanobis_distance(features[i], others)
        distances.append(d)
    distances = np.array(distances)
    anomaly_mask = distances > threshold
    return {
        "anomalies": np.where(anomaly_mask)[0].tolist(),
        "distances": distances.tolist(),
        "mean_distance": float(np.mean(distances)),
        "threshold": threshold,
    }


def anomaly_detect_annotations(annotations: list) -> dict:
    features = []
    valid_indices = []
    for i, ann in enumerate(annotations):
        stats = ann.color_stats
        if not stats or "mean_rgb" not in stats:
            continue
        mean_rgb = stats["mean_rgb"]
        std_rgb = stats["std_rgb"]
        brightness = stats.get("brightness_mean", 0)
        saturation = stats.get("saturation_mean", 0)
        hsv = stats.get("hsv", {})
        hue_mean = hsv.get("hue_mean", 0)
        hue_std = hsv.get("hue_std", 0)
        sat_hsv_mean = hsv.get("saturation_hsv_mean", 0)
        value_mean = hsv.get("value_mean", 0)
        features.append(mean_rgb + std_rgb + [brightness, saturation,
                          hue_mean, hue_std, sat_hsv_mean, value_mean])
        valid_indices.append(i)

    if len(features) < 2:
        return {
            "method": "insufficient_data",
            "anomalies": [],
            "details": {"message": "标注数量不足，无法进行异常检测"},
        }

    features_arr = np.array(features)
    z_result = _z_score_detect(features_arr)
    iqr_result = _iqr_detect(features_arr)
    mah_result = _mahalanobis_detect(features_arr)

    anomaly_counts = np.zeros(len(features))
    for idx in z_result["anomalies"]:
        anomaly_counts[idx] += 1
    for idx in iqr_result["anomalies"]:
        anomaly_counts[idx] += 1
    for idx in mah_result["anomalies"]:
        anomaly_counts[idx] += 1

    per_annotation = []
    for i, idx in enumerate(valid_indices):
        count = int(anomaly_counts[i])
        if count >= 2:
            level = "ANOMALY"
        elif count == 1:
            level = "SUSPICIOUS"
        else:
            level = "NORMAL"
        per_annotation.append({
            "index": idx,
            "level": level,
            "z_score_anomaly": i in z_result["anomalies"],
            "iqr_anomaly": i in iqr_result["anomalies"],
            "mahalanobis_anomaly": i in mah_result["anomalies"],
            "mahalanobis_distance": round(mah_result["distances"][i], 2) if i < len(mah_result["distances"]) else 0,
        })

    total_anomalies = sum(1 for p in per_annotation if p["level"] == "ANOMALY")
    total_suspicious = sum(1 for p in per_annotation if p["level"] == "SUSPICIOUS")
    total_normal = sum(1 for p in per_annotation if p["level"] == "NORMAL")

    n = len(per_annotation)
    if n < 3:
        confidence = "low"
        confidence_note = f"仅 {n} 个有效样本，结果仅供参考，建议增加标注数量"
    elif n < 8:
        confidence = "medium"
        confidence_note = f"{n} 个样本，统计置信度中等"
    else:
        confidence = "high"
        confidence_note = f"{n} 个样本，统计结果可靠"

    return {
        "method": "z_score + iqr + mahalanobis",
        "total": n,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "anomaly_count": total_anomalies,
        "suspicious_count": total_suspicious,
        "normal_count": total_normal,
        "per_annotation": per_annotation,
        "z_score": {"mean": z_result["mean"], "std": z_result["std"], "threshold": cfg("z_score_threshold", 2.5)},
        "iqr": {"q1": iqr_result["q1"], "q3": iqr_result["q3"], "iqr": iqr_result["iqr"]},
        "mahalanobis": {"mean_distance": round(mah_result["mean_distance"], 2), "threshold": cfg("mahalanobis_threshold", 5.0)},
    }


def validate_annotation(image: Image.Image, ann) -> dict:
    stats = ann.color_stats
    if not stats or "error" in stats:
        stats = compute_color_stats(image, ann.bbox)
        ann.color_stats = stats

    issues = []
    score = 1.0

    if stats.get("pixel_count", 0) < 50:
        issues.append("区域像素过少，颜色统计不可靠")
        score -= 0.3

    brightness_std = stats.get("brightness_std", 0)
    if brightness_std > 0.08:
        issues.append(f"亮度方差过大 ({brightness_std:.3f})，可能存在阴影或高光")
        score -= 0.2

    sat_mean = stats.get("saturation_mean", 0)
    if sat_mean < 0.05:
        issues.append(f"饱和度过低 ({sat_mean:.3f})，可能是灰度区域或背景")
        score -= 0.2

    dominants = stats.get("dominant_colors", [])
    if dominants and dominants[0]["ratio"] > 0.85:
        issues.append("颜色过于单一，可能误标了大面积背景")
        score -= 0.15

    score = max(0.0, min(1.0, score))

    if score >= 0.8:
        verdict = "PASS"
    elif score >= 0.5:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    ann.color_verified = True
    return {
        "score": round(score, 2),
        "verdict": verdict,
        "issues": issues,
    }


def validate_all_annotations(image: Image.Image, annotations: list) -> dict:
    results = []
    for ann in annotations:
        r = validate_annotation(image, ann)
        results.append(r)

    anomaly_result = anomaly_detect_annotations(annotations)
    for i, ann_result in enumerate(results):
        if i < len(anomaly_result["per_annotation"]):
            ad = anomaly_result["per_annotation"][i]
            if ad["level"] == "ANOMALY":
                ann_result["score"] = max(0.0, ann_result["score"] - 0.3)
                ann_result["issues"].append(f"异常检测: 三重算法判定为异常 (马氏距离={ad['mahalanobis_distance']})")
                ann_result["verdict"] = "FAIL"
            elif ad["level"] == "SUSPICIOUS":
                ann_result["score"] = max(0.0, ann_result["score"] - 0.15)
                ann_result["issues"].append(f"异常检测: 部分算法判定为可疑")
                if ann_result["verdict"] == "PASS":
                    ann_result["verdict"] = "WARN"

    scores = [r["score"] for r in results]
    verdicts = [r["verdict"] for r in results]

    return {
        "per_annotation": results,
        "avg_score": round(float(np.mean(scores)), 2) if scores else 0.0,
        "pass_count": verdicts.count("PASS"),
        "warn_count": verdicts.count("WARN"),
        "fail_count": verdicts.count("FAIL"),
        "total": len(results),
        "anomaly_detection": anomaly_result,
    }


def compute_cross_image_consistency(project) -> dict:
    label_features = {}
    label_images = {}

    for image_name, anns in project.image_annotations.items():
        for ann in anns:
            stats = ann.color_stats
            if not stats or "mean_rgb" not in stats:
                continue
            label = ann.label
            if label not in label_features:
                label_features[label] = []
                label_images[label] = []
            hsv = stats.get("hsv", {})
            feat = stats["mean_rgb"] + stats["std_rgb"] + [
                stats.get("brightness_mean", 0),
                stats.get("saturation_mean", 0),
                hsv.get("hue_mean", 0),
                hsv.get("hue_std", 0),
                hsv.get("saturation_hsv_mean", 0),
                hsv.get("value_mean", 0),
            ]
            label_features[label].append(feat)
            label_images[label].append(image_name)

    results = {}
    for label in label_features:
        feats = np.array(label_features[label])
        n = len(feats)

        if n < 2:
            results[label] = {
                "count": n,
                "status": "样本不足",
                "consistency_score": None,
                "details": {},
            }
            continue

        mean_all = np.mean(feats, axis=0)
        std_all = np.std(feats, axis=0)
        cv_per_feature = std_all / np.where(mean_all == 0, 1e-6, np.abs(mean_all))
        avg_cv = float(np.mean(cv_per_feature))

        z_result = _z_score_detect(feats)
        iqr_result = _iqr_detect(feats)

        anomaly_indices = set(z_result["anomalies"]) | set(iqr_result["anomalies"])
        anomaly_images = list(set(label_images[label][i] for i in anomaly_indices if i < len(label_images[label])))

        if avg_cv < 0.15:
            status = "高度一致"
            consistency_score = round(1.0 - avg_cv, 3)
        elif avg_cv < 0.35:
            status = "中等一致"
            consistency_score = round(1.0 - avg_cv, 3)
        else:
            status = "差异较大"
            consistency_score = round(max(0.0, 1.0 - avg_cv), 3)

        results[label] = {
            "count": n,
            "status": status,
            "confidence": "low" if n < 5 else ("medium" if n < 10 else "high"),
            "consistency_score": consistency_score,
            "cv_mean": round(avg_cv, 4),
            "mean_rgb": [round(v, 1) for v in mean_all[:3]],
            "std_rgb": [round(v, 1) for v in std_all[:3]],
            "anomaly_count": len(anomaly_indices),
            "anomaly_images": anomaly_images,
            "images_count": len(set(label_images[label])),
        }

    return results


def generate_corrections(image: Image.Image, ann, all_annotations: list = None) -> list:
    corrections = []
    stats = ann.color_stats
    if not stats or "mean_rgb" not in stats:
        return corrections

    x1, y1, x2, y2 = ann.bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    area = bbox_w * bbox_h
    iw, ih = image.size

    if stats.get("pixel_count", 0) < 30:
        corrections.append({
            "type": "SIZE",
            "severity": "HIGH",
            "message": "标注区域过小，像素不足 30，建议扩大 bbox 范围或删除此标注",
            "action": "expand_or_delete",
        })

    if area < 0.001:
        corrections.append({
            "type": "SIZE",
            "severity": "MEDIUM",
            "message": f"bbox 面积仅 {area:.6f}（占图像 {area*100:.3f}%），可能标注了噪声区域",
            "action": "review_size",
        })

    if area > 0.8:
        corrections.append({
            "type": "SIZE",
            "severity": "HIGH",
            "message": f"bbox 面积 {area*100:.1f}% 超过图像 80%，可能是误标整张图或背景",
            "action": "shrink",
        })

    if bbox_w < 0.01 and bbox_h > 0.1:
        corrections.append({
            "type": "ASPECT",
            "severity": "MEDIUM",
            "message": "bbox 极度细长（宽 < 1%），可能是误拖了一条线",
            "action": "redraw",
        })
    if bbox_h < 0.01 and bbox_w > 0.1:
        corrections.append({
            "type": "ASPECT",
            "severity": "MEDIUM",
            "message": "bbox 极度扁平（高 < 1%），可能是误拖了一条线",
            "action": "redraw",
        })

    if stats.get("saturation_mean", 0) < 0.02 and stats.get("brightness_std", 0) < 0.02:
        corrections.append({
            "type": "CONTENT",
            "severity": "HIGH",
            "message": "区域几乎为纯色（低饱和度+低亮度方差），可能是误标了空白/纯色背景",
            "action": "delete",
        })

    dom = stats.get("dominant_colors", [])
    if dom and dom[0]["ratio"] > 0.95:
        corrections.append({
            "type": "CONTENT",
            "severity": "MEDIUM",
            "message": f"区域 95% 以上为单一颜色 ({dom[0]['hex']})，很可能是背景区域",
            "action": "review",
        })

    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        corrections.append({
            "type": "COORDS",
            "severity": "HIGH",
            "message": f"bbox 坐标超出 [0,1] 范围: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]",
            "action": "clamp",
        })

    if all_annotations and len(all_annotations) > 1:
        for i, other in enumerate(all_annotations):
            if other is ann:
                continue
            ox1, oy1, ox2, oy2 = other.bbox
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = area + (ox2 - ox1) * (oy2 - oy1) - intersection
                iou = intersection / union if union > 0 else 0
                if iou > 0.7:
                    corrections.append({
                        "type": "OVERLAP",
                        "severity": "HIGH",
                        "message": f"与标注 {i+1} ({other.label}) IoU={iou:.2f} 严重重叠，建议合并或删除其一",
                        "action": "merge_or_delete",
                        "overlap_index": i,
                        "iou": round(iou, 3),
                    })
                elif iou > 0.3:
                    corrections.append({
                        "type": "OVERLAP",
                        "severity": "MEDIUM",
                        "message": f"与标注 {i+1} ({other.label}) IoU={iou:.2f} 部分重叠",
                        "action": "review_overlap",
                        "overlap_index": i,
                        "iou": round(iou, 3),
                    })

    if not corrections:
        corrections.append({
            "type": "OK",
            "severity": "NONE",
            "message": "未发现明显问题",
            "action": "none",
        })

    return corrections


def generate_batch_corrections(project, image_name: str, image: Image.Image) -> dict:
    anns = project.get_annotations(image_name)
    if not anns:
        return {"corrections": [], "summary": "无标注"}

    all_corrections = []
    for ann in anns:
        if not ann.color_stats or "mean_rgb" not in ann.color_stats:
            ann.color_stats = compute_color_stats(image, ann.bbox)
        corrections = generate_corrections(image, ann, anns)
        all_corrections.append(corrections)

    high_count = sum(1 for cs in all_corrections for c in cs if c["severity"] == "HIGH")
    medium_count = sum(1 for cs in all_corrections for c in cs if c["severity"] == "MEDIUM")

    return {
        "image": image_name,
        "per_annotation": all_corrections,
        "high_count": high_count,
        "medium_count": medium_count,
        "total_annotations": len(anns),
    }
