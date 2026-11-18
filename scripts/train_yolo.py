# -*- coding: utf-8 -*-
"""YOLO26n-seg 对照基线：训练、验证、测试。"""

import argparse
import json
import os

os.environ["YOLO_VERBOSE"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "6")
import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
from ultralytics import YOLO

ROOT = "E:/workspace/2026-11-16-11-53-09/diatom"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", default=os.path.join(ROOT, "models/yolo26n-seg.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data.yaml"))
    ap.add_argument("--eval-data", default=os.path.join(ROOT, "data.yaml"))
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data,
                epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device="cpu", workers=2, seed=42, patience=1000,
                project=os.path.join(ROOT, "outputs/yolo"),
                name="diatom", exist_ok=True, plots=True,
                val=True, verbose=False)

    best = os.path.join(ROOT, "outputs/yolo/diatom/weights/best.pt")
    trained = YOLO(best if os.path.exists(best) else args.weights)
    report = {}
    for split in ("val", "test"):
        m = trained.val(data=args.eval_data, split=split,
                        imgsz=args.imgsz, batch=args.batch, device="cpu",
                        project=os.path.join(ROOT, "outputs/yolo"),
                        name=f"diatom_{split}", verbose=False)
        report[split] = {
            "box_map50_95": float(m.box.map),
            "box_map50": float(m.box.map50),
            "box_precision": float(m.box.mp),
            "box_recall": float(m.box.mr),
            "mask_map50_95": float(m.seg.map),
            "mask_map50": float(m.seg.map50),
            "mask_precision": float(m.seg.mp),
            "mask_recall": float(m.seg.mr),
        }
        print(split, report[split], flush=True)
    with open(os.path.join(ROOT, "outputs/yolo_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("best weights:", best, flush=True)


if __name__ == "__main__":
    main()
