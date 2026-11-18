# -*- coding: utf-8 -*-
"""自建单主干三分支网络在硅藻数据上的训练入口。

复用 YOLO26-PSOS 会话最终生成的 scale_occ_seg 网络：
stride-4 高分辨率层、尺度解耦检测头、高斯感受野正样本分配、
尺度反比损失、可见掩膜与遮挡区双头。硅藻图像没有遮挡区与镜面
高光标注，两个分支保留结构但关闭监督（seg_occ / seg_hl 权重为 0）。
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

ROOT = "E:/workspace/2026-11-16-11-53-09/diatom"
sys.path.insert(0, os.path.join(ROOT, "scale_occ_seg"))
from models.network import ScaleOccSegNet
from datasets.coco_seg_det import CocoSegDetDataset, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--max-train", type=int, default=240)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/self"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.set_num_threads(4)
    train_set = CocoSegDetDataset(
        os.path.join(ROOT, "coco/train.json"),
        os.path.join(ROOT, "data_cache/images"),
        img_size=args.img_size, occ_mode="none")
    if args.max_train > 0 and args.max_train < len(train_set):
        train_set = Subset(train_set, list(range(args.max_train)))
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)
    print(f"train images: {len(train_set)}, steps/epoch: {len(loader)}",
          flush=True)

    model = ScaleOccSegNet(num_classes=1, assign_topk=4,
                           assign_thr=0.5).to("cpu").train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=5e-4)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total, 1), pct_start=0.08)
    weights = {"det_cls": 1.0, "det_box": 2.0, "det_iou": 1.0,
               "seg_vis": 1.0, "seg_occ": 0.0, "seg_bound": 1.0,
               "seg_hl": 0.0}
    log, step, best = [], 0, float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        for batch in loader:
            loss_dict = model.forward_train(
                batch["images"],
                gt_boxes=batch["gt_boxes"],
                gt_labels=batch["gt_labels"],
                vis_masks=batch["vis_masks"],
                occ_masks=batch["occ_masks"],
                hl_masks=batch["hl_masks"])
            loss = sum(weights.get(k, 1.0) * v
                       for k, v in loss_dict.items() if torch.is_tensor(v))
            if not torch.isfinite(loss):
                print("non-finite loss, skip", step, flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0:
                lv = float(loss)
                print(f"ep{epoch} step{step}/{total} loss={lv:.3f} "
                      f"det={float(loss_dict['det_cls']):.3f} "
                      f"seg={float(loss_dict.get('seg_vis', 0)):.3f}",
                      flush=True)
                log.append({"epoch": epoch, "step": step, "loss": lv})
                if lv < best:
                    best = lv
                    torch.save({"model": model.state_dict(), "epoch": epoch,
                                "step": step},
                               os.path.join(args.out, "best.pth"))
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(args.out, "last.pth"))
        print(f"epoch {epoch} done in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out, "train_log.json"), "w",
              encoding="utf-8") as f:
        json.dump({"best_loss": best, "steps": step, "log": log,
                   "epochs": args.epochs, "train_images": len(train_set)},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
