"""PASCAL VOC loading, letterbox preprocessing and training augmentation."""
import pickle
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C

SPLITS = {
    "trainval": [("VOC2007", "trainval"), ("VOC2012", "trainval")],  # 16,551 images
    "test": [("VOC2007", "test")],                                   # 4,952 images
}
VAL_FRACTION, VAL_SEED = 0.10, 0


def load_split(split):
    """'train' / 'val': VOC07+12 trainval split 90/10 with a fixed seed (Report 3: model selection on a validation
    set, test set only for the final number). 'trainval' and 'test': the full official lists.
    The validation image ids are saved in DATA_ROOT/splits/val.txt so the split is reproducible on any machine."""
    if split == "calib":
        return load_calib()
    if split not in ("train", "val"):
        return load_annotations(split)
    items = load_annotations("trainval")
    path = C.DATA_ROOT / "splits" / "val.txt"
    if not path.exists():
        ids = [it["id"] for it in items]
        pick = np.random.default_rng(VAL_SEED).permutation(len(ids))[:round(VAL_FRACTION * len(ids))]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sorted(ids[i] for i in pick)) + "\n")
    val = set(path.read_text().split())
    return [it for it in items if (it["id"] in val) == (split == "val")]


def load_calib():
    """512 calibration images (Report 3.1, 4.1 step 3), drawn with a fixed seed from the training split only
    (never validation or test). Ids are saved in DATA_ROOT/splits/calib.txt, like val.txt."""
    items = load_split("train")
    path = C.DATA_ROOT / "splits" / "calib.txt"
    if not path.exists():
        ids = sorted(it["id"] for it in items)
        pick = np.random.default_rng(C.CALIB_SEED).permutation(len(ids))[:C.CALIB_IMAGES]
        path.write_text("\n".join(sorted(ids[i] for i in pick)) + "\n")
    calib = set(path.read_text().split())
    out = [it for it in items if it["id"] in calib]
    assert len(out) == C.CALIB_IMAGES, f"calib.txt has {len(out)} training images, expected {C.CALIB_IMAGES}"
    return out


def load_annotations(split):
    """List of dicts: path, boxes [n,4] xyxy in original pixels, cls [n], difficult [n]. Cached on disk."""
    cache = C.VOC_ROOT / f"cache_{split}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    items = []
    for year, name in SPLITS[split]:
        root = C.VOC_ROOT / year
        for img_id in (root / "ImageSets" / "Main" / f"{name}.txt").read_text().split():
            boxes, cls, diff = [], [], []
            for obj in ET.parse(root / "Annotations" / f"{img_id}.xml").getroot().iter("object"):
                bb = obj.find("bndbox")
                # VOC pixel coordinates are 1-based
                boxes.append([float(bb.find(k).text) - 1 for k in ("xmin", "ymin", "xmax", "ymax")])
                cls.append(C.CLASSES.index(obj.find("name").text.strip()))
                diff.append(int(obj.find("difficult").text))
            items.append({
                "id": img_id,
                "path": str(root / "JPEGImages" / f"{img_id}.jpg"),
                "boxes": np.array(boxes, np.float32).reshape(-1, 4),
                "cls": np.array(cls, np.int64),
                "difficult": np.array(diff, bool),
            })
    cache.write_bytes(pickle.dumps(items))
    return items


def letterbox(img, size=C.IMG_SIZE, pad=C.PAD_VALUE):
    """Resize keeping the aspect ratio, then pad to size x size (FR3).

    Returns the padded image and (scale, dx, dy) so boxes can be mapped back:
    x_original = (x_letterbox - dx) / scale.
    """
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), pad, np.uint8)
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (r, dx, dy)


def to_tensor(img_bgr):
    """uint8 BGR HxWx3 -> float RGB 3xHxW in [0,1]. The only input normalisation is x/255."""
    return torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1].transpose(2, 0, 1))).float() / 255


def _augment(img, boxes, cls, size):
    """Random scale + placement, horizontal flip and HSV jitter."""
    h, w = img.shape[:2]
    r = size / max(h, w) * random.uniform(0.5, 1.5)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # Random offset: slides inside the canvas if smaller, crops if larger.
    dx = random.randint(min(0, size - nw), max(0, size - nw))
    dy = random.randint(min(0, size - nh), max(0, size - nh))
    out = np.full((size, size, 3), C.PAD_VALUE, np.uint8)
    sx, sy = max(-dx, 0), max(-dy, 0)
    tx, ty = max(dx, 0), max(dy, 0)
    cw, ch = min(nw - sx, size - tx), min(nh - sy, size - ty)
    out[ty:ty + ch, tx:tx + cw] = img[sy:sy + ch, sx:sx + cw]

    boxes = boxes * r + np.array([dx, dy, dx, dy], np.float32)
    area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxes = boxes.clip(0, size)
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (bw > 2) & (bh > 2) & (bw * bh > 0.25 * area0)  # drop boxes mostly cropped away
    boxes, cls = boxes[keep], cls[keep]

    if random.random() < 0.5:
        out = out[:, ::-1]
        boxes[:, [0, 2]] = size - boxes[:, [2, 0]]

    hsv = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-0.015, 0.015) * 180) % 180
    hsv[..., 1] = (hsv[..., 1] * random.uniform(0.3, 1.7)).clip(0, 255)
    hsv[..., 2] = (hsv[..., 2] * random.uniform(0.6, 1.4)).clip(0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), boxes, cls


class VOCDataset(Dataset):
    def __init__(self, split, augment=False, size=C.IMG_SIZE):
        self.items, self.augment, self.size = load_split(split), augment, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(it["path"])
        if self.augment:
            keep = ~it["difficult"]  # darknet's VOC recipe also skips 'difficult' objects in training
            img, boxes, cls = _augment(img, it["boxes"][keep], it["cls"][keep], self.size)
        else:
            img, (r, dx, dy) = letterbox(img, self.size)
            boxes, cls = it["boxes"] * r + np.array([dx, dy, dx, dy], np.float32), it["cls"]
        # targets: class, cx, cy, w, h normalised to [0,1]
        t = np.zeros((len(boxes), 5), np.float32)
        if len(boxes):
            t[:, 0] = cls
            t[:, 1:3] = (boxes[:, :2] + boxes[:, 2:]) / 2 / self.size
            t[:, 3:5] = (boxes[:, 2:] - boxes[:, :2]) / self.size
        return to_tensor(img), torch.from_numpy(t), i


def collate(batch):
    imgs, targets, idx = zip(*batch)
    targets = [torch.cat((torch.full((len(t), 1), b, dtype=torch.float32), t), 1) for b, t in enumerate(targets)]
    return torch.stack(imgs), torch.cat(targets), torch.tensor(idx)
