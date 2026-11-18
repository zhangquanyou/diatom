# -*- coding: utf-8 -*-
"""硅藻数据集转换。

输入  data/images/*.png, data/annotations/*.png, data/xmls/*.xml
输出  降采样图像缓存、COCO 实例分割标注、YOLO-seg 多边形标注、划分文件。

所有 68 个硅藻物种合并为单类 diatom，物种名保留在 COCO 标注的 species 字段。
未被任何框匹配的掩膜连通域视为杂质或背景，不产生正样本。
"""

import glob
import json
import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = "E:/workspace/2026-11-16-11-53-09"
SRC = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "diatom")
CACHE = os.path.join(OUT, "data_cache")
CW, CH = 1024, 768
SEED = 42
MIN_COMPONENT_AREA = 200

CAT = [{"id": 1, "name": "diatom", "supercategory": "alga"}]


def read_boxes(path):
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall(".//object"):
        b = obj.find("bbox")
        boxes.append({
            "xyxy": [float(b.findtext("xmin")), float(b.findtext("ymin")),
                     float(b.findtext("xmax")), float(b.findtext("ymax"))],
            "species": obj.findtext("name", "unknown"),
        })
    return boxes


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(ua + ub - inter, 1e-6)


def get_components(mask):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        (mask > 127).astype("uint8"), 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_COMPONENT_AREA:
            continue
        comps.append({"idx": i, "xyxy": [float(x), float(y),
                                         float(x + w), float(y + h)],
                      "area": int(area)})
    return labels, comps


def match_instances(boxes, comps, labels):
    pairs = []
    for bi, b in enumerate(boxes):
        for ci, c in enumerate(comps):
            v = box_iou(b["xyxy"], c["xyxy"])
            if v > 0.02:
                pairs.append((v, bi, ci))
    pairs.sort(reverse=True)
    assign, used_c = {}, set()
    for _, bi, ci in pairs:
        if bi in assign or ci in used_c:
            continue
        assign[bi] = ci
        used_c.add(ci)
    for bi, b in enumerate(boxes):
        if bi in assign:
            continue
        cx = int((b["xyxy"][0] + b["xyxy"][2]) / 2)
        cy = int((b["xyxy"][1] + b["xyxy"][3]) / 2)
        if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
            lab = int(labels[cy, cx])
            for ci, c in enumerate(comps):
                if c["idx"] == lab and ci not in used_c:
                    assign[bi] = ci
                    used_c.add(ci)
                    break
    return assign


def polygon_of(labels, comp, sx, sy, fallback_box=None):
    if comp is not None:
        m = (labels == comp["idx"]).astype("uint8")
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            p = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
            if len(p) >= 3:
                return np.stack([p[:, 0] * sx, p[:, 1] * sy], axis=1)
    x1, y1, x2, y2 = fallback_box
    return np.array([[x1 * sx, y1 * sy], [x2 * sx, y1 * sy],
                     [x2 * sx, y2 * sy], [x1 * sx, y2 * sy]],
                    dtype=np.float32)


def main():
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(SRC, "xmls", "*.xml")))
    stems = [s for s in stems if s.isdigit()]
    empty = [s for s in stems
             if not read_boxes(os.path.join(SRC, "xmls", s + ".xml"))]
    nonempty = [s for s in stems if s not in set(empty)]
    rng = random.Random(SEED)
    rng.shuffle(nonempty)
    rng.shuffle(empty)

    n = len(stems)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    split = {}
    split["train"] = nonempty[:n_tr]
    split["val"] = nonempty[n_tr:n_tr + n_va]
    split["test"] = nonempty[n_tr + n_va:]
    for i, s in enumerate(empty):
        split["train" if i < 9 else ("val" if i < 11 else "test")].append(s)
    for k in split:
        rng.shuffle(split[k])

    coco = {k: {"images": [], "annotations": [], "categories": CAT}
            for k in split}
    stats = {k: {"images": 0, "empty": 0, "instances": 0,
                 "mask_fallback": 0, "background_components": 0}
             for k in split}
    ann_id = 0
    sx, sy = CW / 2112.0, CH / 1584.0

    for sp, items in split.items():
        img_dir = os.path.join(CACHE, "images", sp)
        lab_dir = os.path.join(CACHE, "labels", sp)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for stem in items:
            img = cv2.imread(os.path.join(SRC, "images", stem + ".png"))
            mask = cv2.imread(os.path.join(SRC, "annotations", stem + ".png"),
                              cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print("missing", stem)
                continue
            oh, ow = mask.shape[:2]
            sx, sy = CW / float(ow), CH / float(oh)
            small = cv2.resize(img, (CW, CH), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            boxes = read_boxes(os.path.join(SRC, "xmls", stem + ".xml"))
            labels, comps = get_components(mask)
            assign = match_instances(boxes, comps, labels) if boxes else {}

            iid = int(stem)
            coco[sp]["images"].append({
                "id": iid, "file_name": f"{sp}/{stem}.jpg",
                "width": CW, "height": CH})
            stats[sp]["images"] += 1
            if not boxes:
                stats[sp]["empty"] += 1
            stats[sp]["background_components"] += len(comps) - len(assign)

            yolo_lines = []
            for bi, b in enumerate(boxes):
                comp = comps[assign[bi]] if bi in assign else None
                if comp is None:
                    stats[sp]["mask_fallback"] += 1
                poly = polygon_of(labels, comp, sx, sy, b["xyxy"])
                poly[:, 0] = np.clip(poly[:, 0], 0, CW - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, CH - 1)
                x1, y1, x2, y2 = [v for v in b["xyxy"]]
                x1, x2 = min(x1, x2) * sx, max(x1, x2) * sx
                y1, y2 = min(y1, y2) * sy, max(y1, y2) * sy
                bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
                flat = [round(float(v), 2) for v in poly.reshape(-1)]
                ann_id += 1
                coco[sp]["annotations"].append({
                    "id": ann_id, "image_id": iid, "category_id": 1,
                    "bbox": [round(x1, 2), round(y1, 2),
                             round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "iscrowd": 0,
                    "segmentation": [flat], "species": b["species"]})
                norm = []
                for j, v in enumerate(flat):
                    norm.append(round(v / (CW if j % 2 == 0 else CH), 6))
                yolo_lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
                stats[sp]["instances"] += 1
            with open(os.path.join(lab_dir, stem + ".txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
        print(sp, "done", stats[sp])

    for sp in split:
        with open(os.path.join(OUT, "coco", sp + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(coco[sp], f, ensure_ascii=False)
    with open(os.path.join(OUT, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {CACHE}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\nnames: {0: diatom}\n")
    with open(os.path.join(OUT, "prepare_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("total", stats)


if __name__ == "__main__":
    main()
