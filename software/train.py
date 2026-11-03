"""Train the Float32 YOLOv3-Tiny baseline at 320x320.

Trains on 90% of VOC07+12 trainval, picks the best epoch on the other 10% (validation), then evaluates that
frozen model once on the VOC2007 test set (Report 3, Sections 4.1 and 4.6).

Usage: python train.py --name float32-v2 [--epochs 60] [--batch 64]
Mixed precision and TF32 are off on purpose: Report 2 (5.2, IEEE 754) requires pure Float32 reference runs.
"""
import argparse
import copy
import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from data import VOCDataset, collate
from evaluate import evaluate
from loss import compute_loss
from model import YoloV3Tiny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="float32")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--pretrained", default=str(C.WEIGHTS_DIR / "yolov3-tiny.weights"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False  # true IEEE Float32
    device = torch.device("cuda")
    out = C.RUNS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({**vars(args), "torch": torch.__version__, "leaky": C.LEAKY_SLOPE, "anchors": C.ANCHORS,
                                               "tf32": False}, indent=1))

    model = YoloV3Tiny().to(device)
    if args.pretrained and args.pretrained != "none":
        model.load_darknet(args.pretrained)  # COCO weights, heads cut down to the 20 VOC classes
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]  # biases and BatchNorm
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0}],
                          lr=args.lr, momentum=0.937, nesterov=True)
    loader = DataLoader(VOCDataset("train", augment=True), args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=True, persistent_workers=True)
    warmup, total = 3 * len(loader), args.epochs * len(loader)
    lr_at = lambda it: args.lr * (it / warmup if it < warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - warmup) / (total - warmup))))

    meta = {"leaky": C.LEAKY_SLOPE, "anchors": C.ANCHORS}  # saved in every checkpoint
    it, best = 0, 0.0
    log = open(out / "log.csv", "a")
    for epoch in range(args.epochs):
        model.train()
        t0, sums = time.time(), torch.zeros(3, device=device)
        for imgs, targets, _ in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            loss, parts = compute_loss(model(imgs.to(device, non_blocking=True)), targets.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            it += 1
            d = min(0.9999, (1 + it) / (10 + it))
            for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
                e.mul_(d).add_(m.detach(), alpha=1 - d) if e.dtype.is_floating_point else e.copy_(m)
            sums += parts
        box, obj, cls = (sums / len(loader)).tolist()
        line = f"{epoch + 1},{box:.4f},{obj:.4f},{cls:.4f},{time.time() - t0:.0f}"
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            aps = evaluate(ema, device, split="val")
            m11, mall = aps[:, 0].mean() * 100, aps[:, 1].mean() * 100
            line += f",{m11:.2f},{mall:.2f}"
            if m11 > best:
                best = m11
                torch.save({"model": ema.state_dict(), "epoch": epoch + 1, "val_map11": m11, "val_map": mall, **meta}, out / "best.pt")
        torch.save({"model": ema.state_dict(), "raw": model.state_dict(), "epoch": epoch + 1, **meta}, out / "last.pt")
        print(line, flush=True)
        log.write(line + "\n"); log.flush()
    # The chosen model is now frozen; evaluate it once on the test set.
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    aps = evaluate(model, device, split="test")
    result = {"best_epoch": ck["epoch"], "val_map11": ck["val_map11"], "val_map": ck["val_map"],
              "test_map11": aps[:, 0].mean() * 100, "test_map": aps[:, 1].mean() * 100}
    (out / "final.json").write_text(json.dumps({k: round(float(v), 2) for k, v in result.items()}, indent=1))
    print("final:", result)


if __name__ == "__main__":
    main()
