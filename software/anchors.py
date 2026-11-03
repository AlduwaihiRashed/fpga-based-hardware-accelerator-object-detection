"""K-means anchors for YOLOv3-Tiny on VOC at 320x320 (Report 3, Section 4.1 step 1).

Distance = 1 - IoU between box shapes (both centred at the origin), as in the YOLOv2/v3 papers.
Uses the non-difficult boxes of the training split (not validation, not test), letterboxed to 320.

Usage: python anchors.py      -> prints the 6 anchors to paste into config.py and how well they fit
"""
import numpy as np

import config as C
from data import load_split


def shape_iou(wh, anchors):
    """IoU of every box shape against every anchor shape, [n, k]."""
    inter = np.minimum(wh[:, None, 0], anchors[None, :, 0]) * np.minimum(wh[:, None, 1], anchors[None, :, 1])
    return inter / (wh[:, None].prod(2) + anchors[None].prod(2) - inter)


def kmeans(wh, k=6, seed=0, iters=300):
    rng = np.random.default_rng(seed)
    anchors = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        near = shape_iou(wh, anchors).argmax(1)
        new = np.array([np.median(wh[near == j], 0) if (near == j).any() else anchors[j] for j in range(k)])
        if np.allclose(new, anchors):
            break
        anchors = new
    return anchors[np.argsort(anchors.prod(1))]


def training_box_shapes():
    wh = []
    for it in load_split("train"):
        w, h = __import__("PIL.Image", fromlist=["Image"]).open(it["path"]).size
        r = C.IMG_SIZE / max(w, h)
        b = it["boxes"][~it["difficult"]]
        wh.append(np.stack((b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]), 1) * r)
    return np.concatenate(wh)


if __name__ == "__main__":
    wh = training_box_shapes()
    darknet = np.array([(23, 27), (37, 58), (81, 82), (81, 82), (135, 169), (344, 319)], float)
    # Best of 10 restarts (highest mean IoU), so the result does not depend on one lucky start.
    runs = [kmeans(wh, seed=s) for s in range(10)]
    best = max(runs, key=lambda a: shape_iou(wh, a).max(1).mean())
    print(f"{len(wh)} training boxes")
    print("mean best IoU, darknet anchors:", round(shape_iou(wh, darknet).max(1).mean(), 4))
    print("mean best IoU, k-means anchors:", round(shape_iou(wh, best).max(1).mean(), 4))
    print("ANCHORS_KMEANS =", [tuple(int(round(v)) for v in a) for a in best])
