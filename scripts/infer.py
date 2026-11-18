# -*- coding: utf-8 -*-
"""统一推理接口。

两个模型都返回逐实例的框、分数与掩膜，供计数界面和评测共用。
只有标注为硅藻的实例计入结果，背景与杂散颗粒由置信度与最小面积
两道过滤排除。
"""

import os
import sys

import cv2
import numpy as np
import torch

ROOT = "E:/workspace/2026-11-16-11-53-09/diatom"
sys.path.insert(0, os.path.join(ROOT, "scale_occ_seg"))

SELF_CKPT = os.path.join(ROOT, "outputs/self/best.pth")
YOLO_CKPT = os.path.join(ROOT, "outputs/yolo/diatom/weights/best.pt")
YOLO_BASE = os.path.join(ROOT, "models/yolo26n-seg.pt")


def load_yolo(path=None):
    from ultralytics import YOLO
    p = path or (YOLO_CKPT if os.path.exists(YOLO_CKPT) else YOLO_BASE)
    return YOLO(p), os.path.basename(p)


def load_self(path=None):
    from models.network import ScaleOccSegNet
    p = path or SELF_CKPT
    model = ScaleOccSegNet(num_classes=1, assign_topk=4, assign_thr=0.5)
    if os.path.exists(p):
        ck = torch.load(p, map_location="cpu")
        model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    return model, os.path.basename(p)


def _empty(h, w):
    return {"boxes": [], "scores": [], "masks": [], "count": 0}


def _filter(result, min_area, h, w, max_instances=20):
    boxes, scores, masks = [], [], []
    order = np.argsort(np.asarray(result["scores"]))[::-1][:max_instances]
    for i in order:
        box, sc, m = result["boxes"][i], result["scores"][i], result["masks"][i]
        if int(m.sum()) < min_area:
            continue
        boxes.append([float(max(0, min(w, v))) if j % 2 == 0 else
                      float(max(0, min(h, v)))
                      for j, v in enumerate(box)])
        scores.append(float(sc))
        masks.append(np.asarray(m).astype(np.uint8))
    return {"boxes": boxes, "scores": scores, "masks": masks,
            "count": len(boxes)}


def predict_yolo(model, bgr, conf=0.30, imgsz=480, min_area=120,
                 max_instances=20):
    h, w = bgr.shape[:2]
    res = model.predict(bgr, imgsz=imgsz, conf=conf, device="cpu",
                        retina_masks=True, verbose=False)[0]
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        return _empty(h, w)
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    raw = res.masks.data.cpu().numpy()
    out = {"boxes": [], "scores": [], "masks": []}
    for i in range(len(boxes)):
        m = cv2.resize(raw[i].astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST)
        out["boxes"].append(boxes[i].tolist())
        out["scores"].append(float(scores[i]))
        out["masks"].append((m > 0.5).astype(np.uint8))
    return _filter(out, min_area, h, w, max_instances)


def _resize_for_self(bgr, img_size):
    h, w = bgr.shape[:2]
    scale = img_size / min(h, w)
    nw = max(32, int(w * scale) // 32 * 32)
    nh = max(32, int(h * scale) // 32 * 32)
    small = cv2.resize(bgr, (nw, nh))
    t = torch.from_numpy(
        small[..., ::-1].copy().astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    return t, scale


def predict_self(model, bgr, conf=0.30, img_size=384, min_area=120,
                 max_instances=20):
    h, w = bgr.shape[:2]
    torch.set_num_threads(4)
    x, scale = _resize_for_self(bgr, img_size)
    with torch.no_grad():
        r = model.forward_infer(x, score_thr=conf, nms_iou=0.5)[0]
    if len(r["boxes"]) == 0:
        return _empty(h, w)
    vis = r["vis_prob"][0].cpu().numpy() if hasattr(r["vis_prob"], "cpu") \
        else np.asarray(r["vis_prob"][0])
    boxes = r["boxes"].cpu().numpy()
    scores = r["scores"].cpu().numpy()
    out = {"boxes": [], "scores": [], "masks": []}
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x.shape[3], x2), min(x.shape[2], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        small_m = (vis[y1:y2, x1:x2] > 0.5).astype(np.uint8)
        if small_m.sum() == 0:
            continue
        m = cv2.resize(small_m, (w, h), interpolation=cv2.INTER_NEAREST)
        out["boxes"].append([x1 / scale, y1 / scale, x2 / scale, y2 / scale])
        out["scores"].append(float(scores[i]))
        out["masks"].append(m)
    return _filter(out, min_area, h, w, max_instances)


def overlay(bgr, result, title="diatom"):
    img = bgr.copy()
    rng = np.random.default_rng(7)
    for i, (box, sc, m) in enumerate(zip(result["boxes"], result["scores"],
                                         result["masks"])):
        color = tuple(int(v) for v in rng.integers(40, 230, 3))
        layer = img.copy()
        layer[m > 0] = (0.55 * layer[m > 0] + 0.45 * np.array(color)).astype(
            np.uint8)
        img = layer
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{i+1} {sc:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(img, f"{title} count = {result['count']}",
                (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (20, 20, 220), 2)
    return img
