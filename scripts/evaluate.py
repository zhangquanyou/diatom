# -*- coding: utf-8 -*-
"""测试集评测：计数误差、误检漏检、掩膜 IoU、纯背景误检。

除测试集外，额外从测试图裁出不含任何硅藻标注的背景块作为杂质探针，
检验模型是否把非硅藻对象计入结果。
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer import load_yolo, load_self, predict_yolo, predict_self

ROOT = "E:/workspace/2026-11-16-11-53-09/diatom"


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max((a[2] - a[0]) * (a[3] - a[1]), 0.0)
    ub = max((b[2] - b[0]) * (b[3] - b[1]), 0.0)
    return inter / max(ua + ub - inter, 1e-6)


def greedy_match(gt_boxes, pred_boxes, thr=0.5):
    pairs = []
    for pi, p in enumerate(pred_boxes):
        for gi, g in enumerate(gt_boxes):
            v = box_iou(p, g)
            if v >= thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append((pi, gi, v))
    return matched


def mask_iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def background_probes(coco, n=30):
    probes = []
    for im in coco["images"]:
        if len(probes) >= n:
            break
        path = os.path.join(ROOT, "data_cache/images", im["file_name"])
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = [xyxy(a["bbox"]) for a in coco["annotations"]
               if a["image_id"] == im["id"]]
        rng = np.random.default_rng(im["id"])
        for _ in range(20):
            s = 320
            x0 = int(rng.integers(0, max(1, w - s)))
            y0 = int(rng.integers(0, max(1, h - s)))
            q = [x0, y0, x0 + s, y0 + s]
            if any(box_iou(q, g) > 0 for g in gts):
                continue
            probes.append(img[y0:y0 + s, x0:x0 + s])
            break
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolo", "self"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    with open(os.path.join(ROOT, f"coco/{args.split}.json"),
              encoding="utf-8") as f:
        coco = json.load(f)
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    images = coco["images"]
    if args.limit > 0:
        images = images[:args.limit]

    if args.model == "yolo":
        model, ck = load_yolo()
        predict = lambda img: predict_yolo(model, img, conf=args.conf,
                                           imgsz=args.imgsz)
    else:
        model, ck = load_self()
        predict = lambda img: predict_self(model, img, conf=args.conf)

    rows = []
    tp = fp = fn = 0
    ious, count_err = [], []
    empty_pred = 0
    empty_images = 0
    for k, im in enumerate(images):
        path = os.path.join(ROOT, "data_cache/images", im["file_name"])
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        gts = [xyxy(a["bbox"]) for a in by_img.get(im["id"], [])]
        res = predict(bgr)
        matched = greedy_match(gts, res["boxes"])
        tp += len(matched)
        fp += len(res["boxes"]) - len(matched)
        fn += len(gts) - len(matched)
        for pi, gi, _ in matched:
            gt_m = np.zeros((im["height"], im["width"]), np.uint8)
            a = by_img[im["id"]][gi]
            pts = np.array(a["segmentation"][0], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(gt_m, [pts], 1)
            ious.append(mask_iou(res["masks"][pi], gt_m))
        count_err.append(res["count"] - len(gts))
        if not gts:
            empty_images += 1
            empty_pred += res["count"]
        rows.append({"image": im["file_name"], "gt": len(gts),
                     "pred": res["count"], "tp": len(matched),
                     "fp": len(res["boxes"]) - len(matched),
                     "fn": len(gts) - len(matched)})
        if (k + 1) % 25 == 0:
            print(f"evaluated {k+1}/{len(images)}", flush=True)

    probes = background_probes(coco)
    probe_hits = sum(predict(p)["count"] for p in probes)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    report = {
        "model": args.model, "checkpoint": ck, "conf": args.conf,
        "split": args.split,
        "test_images": len(rows), "gt_instances": tp + fn,
        "pred_instances": tp + fp, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-6), 4),
        "count_mae": round(float(np.mean(np.abs(count_err))), 4),
        "count_rmse": round(float(np.sqrt(np.mean(np.square(count_err)))), 4),
        "exact_count_rate": round(float(np.mean(np.array(count_err) == 0)), 4),
        "mask_iou_mean": round(float(np.mean(ious)) if ious else 0.0, 4),
        "empty_images": empty_images,
        "false_positives_on_empty_images": empty_pred,
        "background_probes": len(probes),
        "background_probe_false_positives": int(probe_hits),
        "background_probe_false_positive_rate":
            round(probe_hits / max(len(probes), 1), 4),
    }
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"eval_{args.model}_{args.split}.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"eval_{args.model}_{args.split}.csv"),
              "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
