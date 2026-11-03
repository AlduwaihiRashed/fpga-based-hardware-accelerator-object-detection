"""mAP@0.5 following the VOC protocol, on the validation split (model selection) or VOC2007 test (final number).

Usage: python evaluate.py <checkpoint.pt> [--split test|val]
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from data import VOCDataset, collate
from model import YoloV3Tiny
from yolov3_tiny_float32 import decode


def average_precision(rec, prec):
    """Returns (VOC2007 11-point AP, all-point area-under-curve AP)."""
    ap11 = sum(prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.arange(0, 1.1, 0.1)) / 11
    mrec, mpre = np.concatenate(([0], rec, [1])), np.concatenate(([0], prec, [0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return ap11, ((mrec[i + 1] - mrec[i]) * mpre[i + 1]).sum()


def voc_map(dets, items, iou_thres=0.5, curves=False):
    """dets: per image [n,6] (x1,y1,x2,y2,score,class) in original pixels. 'Difficult' objects are ignored.

    With curves=True also returns, per class, a dict with recall, precision, scores and the object count."""
    aps, pr = [], []
    for c in range(C.NUM_CLASSES):
        gts, npos, rows = {}, 0, []
        for k, it in enumerate(items):
            m = it["cls"] == c
            gts[k] = (it["boxes"][m], it["difficult"][m], np.zeros(m.sum(), bool))
            npos += (~it["difficult"][m]).sum()
            d = dets[k][dets[k][:, 5] == c]
            rows += [(s, k, bx) for *bx, s in d[:, :5].tolist()]
        rows.sort(key=lambda r: -r[0])
        tp, fp = np.zeros(len(rows)), np.zeros(len(rows))
        for n, (_, k, bx) in enumerate(rows):
            g, diff, used = gts[k]
            best, j = 0.0, -1
            if len(g):
                # VOC devkit counts pixels inclusively, hence the +1
                iw = (np.minimum(g[:, 2], bx[2]) - np.maximum(g[:, 0], bx[0]) + 1).clip(0)
                ih = (np.minimum(g[:, 3], bx[3]) - np.maximum(g[:, 1], bx[1]) + 1).clip(0)
                union = (bx[2] - bx[0] + 1) * (bx[3] - bx[1] + 1) + (g[:, 2] - g[:, 0] + 1) * (g[:, 3] - g[:, 1] + 1) - iw * ih
                iou = iw * ih / union
                j = int(iou.argmax())
                best = iou[j]
            if best > iou_thres:
                if diff[j]:
                    continue
                if not used[j]:
                    tp[n], used[j] = 1, True
                else:
                    fp[n] = 1
            else:
                fp[n] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        rec, prec = tp / max(npos, 1), tp / np.maximum(tp + fp, 1e-9)
        aps.append(average_precision(rec, prec))
        pr.append({"recall": rec, "precision": prec, "scores": np.array([r[0] for r in rows]), "npos": int(npos)})
    return (np.array(aps), pr) if curves else np.array(aps)  # aps: [20, 2]


@torch.no_grad()
def predict(model, device, split="test", batch=64, workers=8, forward=None):
    """Run a split and return detections in original-image pixels. `forward` lets quantised models plug in."""
    ds = VOCDataset(split)
    dets = [None] * len(ds)
    model.eval()
    for imgs, _, idx in DataLoader(ds, batch, num_workers=workers, collate_fn=collate):
        preds = (forward or model)(imgs.to(device))
        for d, i in zip(decode(preds), idx.tolist()):
            w, h = _size(ds.items[i])
            r = C.IMG_SIZE / max(h, w)
            dx, dy = (C.IMG_SIZE - round(w * r)) // 2, (C.IMG_SIZE - round(h * r)) // 2
            d = d.cpu().numpy()
            d[:, [0, 2]] = ((d[:, [0, 2]] - dx) / r).clip(0, w - 1)
            d[:, [1, 3]] = ((d[:, [1, 3]] - dy) / r).clip(0, h - 1)
            dets[i] = d
    return dets, ds.items


def _size(item):
    if "size" not in item:
        from PIL import Image
        item["size"] = Image.open(item["path"]).size
    return item["size"]


def evaluate(model, device, **kw):
    dets, items = predict(model, device, **kw)
    return voc_map(dets, items)


def report(aps):
    for name, (a11, a) in zip(C.CLASSES, aps):
        print(f"  {name:12s} AP11 {a11 * 100:5.1f}   AP {a * 100:5.1f}")
    print(f"mAP@0.5  VOC07 11-point: {aps[:, 0].mean() * 100:.2f}   all-point: {aps[:, 1].mean() * 100:.2f}")


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False  # true IEEE Float32
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV3Tiny().to(device)
    model.load_state_dict(torch.load(sys.argv[1], map_location=device, weights_only=False)["model"])
    report(evaluate(model, device, split=split))
