"""YOLOv3-Tiny in Float32, kept simple and ready for INT8. This is the Float32 reference model of Report 3.

Every layer here is just   conv (weights + bias)  ->  Leaky ReLU (slope 0.1)
BatchNorm is folded into the conv weights when the weights are loaded, so this model is
exactly what the FPGA will run, and exactly what we will quantize to INT8 and Posit8.

The file holds the whole Float32 model: the 13 layers, weight loading + BN folding,
and decode() (box decoding + NMS, the part the ARM CPU runs on the board).
Fixed numbers (slope, anchors, thresholds) come from config.py.

Weights: our fine-tuned checkpoint from train.py (runs/<name>/best.pt), or darknet's
yolov3-tiny.weights (COCO), keeping the 20 VOC classes in the two heads.

Usage:
    python yolov3_tiny_float32.py /mnt/ssd/Data/senior/runs/float32-v2/best.pt          # self-check
    python yolov3_tiny_float32.py /mnt/ssd/Data/senior/runs/float32-v2/best.pt --map    # + mAP on VOC2007 test
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

# The 13 convolutions, in darknet's file order: (name, in channels, out channels, kernel size, is head).
# Heads have no BatchNorm and no activation. Output sizes are for a 320x320 input.
LAYERS = [
    ("c0",     3,   16, 3, False),   # 16 x 320 x 320
    ("c2",    16,   32, 3, False),   # 32 x 160 x 160
    ("c4",    32,   64, 3, False),   # 64 x 80 x 80
    ("c6",    64,  128, 3, False),   # 128 x 40 x 40
    ("c8",   128,  256, 3, False),   # 256 x 20 x 20   (kept for the concat)
    ("c10",  256,  512, 3, False),   # 512 x 10 x 10
    ("c12",  512, 1024, 3, False),   # 1024 x 10 x 10
    ("c13", 1024,  256, 1, False),   # 256 x 10 x 10   (feeds both branches)
    ("c14",  256,  512, 3, False),   # 512 x 10 x 10
    ("h15",  512,   75, 1, True),    # 75 x 10 x 10    head, stride 32
    ("c18",  256,  128, 1, False),   # 128 x 10 x 10
    ("c21",  384,  256, 3, False),   # 256 x 20 x 20   (input = upsample(c18) ++ c8)
    ("h22",  256,   75, 1, True),    # 75 x 20 x 20    head, stride 16
]


class YoloV3TinyFloat32(nn.Module):
    def __init__(self, num_classes=C.NUM_CLASSES):
        super().__init__()
        self.nc = num_classes
        self.conv = nn.ModuleDict()
        for name, cin, cout, k, is_head in LAYERS:
            if is_head:
                cout = 3 * (5 + num_classes)
            self.conv[name] = nn.Conv2d(cin, cout, k, stride=1, padding=k // 2, bias=True)

    def layer(self, name, x):
        """One layer: conv + bias, then Leaky ReLU (heads skip the activation).

        This is the only place the arithmetic happens. The INT8 model will subclass this
        class and override just this method (quantize input, weights and output).
        """
        y = self.conv[name](x)
        return y if name.startswith("h") else F.leaky_relu(y, C.LEAKY_SLOPE)

    def forward(self, x):
        """x: (B, 3, 320, 320), RGB letterboxed, scaled to 0..1.
        Returns raw head outputs [(B,3,20,20,25), (B,3,10,10,25)], stride 16 first."""
        pool = lambda t: F.max_pool2d(t, 2, 2)
        x = pool(self.layer("c0", x))
        x = pool(self.layer("c2", x))
        x = pool(self.layer("c4", x))
        x = pool(self.layer("c6", x))
        r8 = self.layer("c8", x)
        x = self.layer("c10", pool(r8))
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode="replicate"), 2, 1)  # layer 11: stride 1, size kept
        r13 = self.layer("c13", self.layer("c12", x))
        out32 = self.layer("h15", self.layer("c14", r13))
        x = F.interpolate(self.layer("c18", r13), scale_factor=2, mode="nearest")
        out16 = self.layer("h22", self.layer("c21", torch.cat((x, r8), 1)))
        return [self._split(out16), self._split(out32)]

    def _split(self, o):
        b, _, h, w = o.shape
        return o.view(b, 3, 5 + self.nc, h, w).permute(0, 1, 3, 4, 2).contiguous()

    # ---------------- loading weights ----------------

    def _set(self, name, weight, bias):
        self.conv[name].weight.data.copy_(weight)
        self.conv[name].bias.data.copy_(bias)

    def _set_folded(self, name, weight, gamma, beta, mean, var, eps=1e-5):
        """Fold BatchNorm into the conv: w' = w * s, b' = beta - mean * s, with s = gamma / sqrt(var + eps)."""
        s = gamma / torch.sqrt(var + eps)
        self._set(name, weight * s.view(-1, 1, 1, 1), beta - mean * s)

    def load_checkpoint(self, path):
        """Load a checkpoint saved by train.py (model.py's layout, with BatchNorm) and fold it."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # Checkpoints from float32-v2 on record the slope and anchors they were trained with.
        if "anchors" in ck:
            assert [list(map(list, a)) for a in ck["anchors"]] == [list(map(list, a)) for a in C.ANCHORS], \
                "checkpoint was trained with different anchors than config.py"
            assert ck["leaky"] == C.LEAKY_SLOPE, "checkpoint was trained with a different Leaky slope"
        sd = ck["model"]
        for name, *_, is_head in LAYERS:
            if is_head:
                self._set(name, sd[f"{name}.weight"], sd[f"{name}.bias"])
            else:
                g = lambda k: sd[f"{name}.bn.{k}"]
                self._set_folded(name, sd[f"{name}.conv.weight"], g("weight"), g("bias"), g("running_mean"), g("running_var"))
        return self

    def load_darknet(self, path):
        """Load darknet's yolov3-tiny.weights (COCO, 80 classes) and keep the 20 VOC classes in the heads."""
        with open(path, "rb") as f:
            major, minor, _ = np.fromfile(f, np.int32, 3)
            np.fromfile(f, np.int64 if major * 10 + minor >= 2 else np.int32, 1)  # "images seen" counter
            buf = torch.from_numpy(np.fromfile(f, np.float32))
        pos = 0

        def take(*shape):
            nonlocal pos
            n = int(np.prod(shape))
            t = buf[pos:pos + n].view(*shape)
            pos += n
            return t

        for name, cin, cout, k, is_head in LAYERS:
            if is_head:
                # darknet stores the bias first, then the weights. Keep box + objectness + the 20 VOC classes per anchor.
                keep = [a * 85 + i for a in range(3) for i in [0, 1, 2, 3, 4] + [5 + c for c in C.COCO_INDEX]]
                bias = take(255)[keep]
                self._set(name, take(255, cin, k, k)[keep], bias)
            else:
                # darknet order: BN beta, gamma, mean, variance, then the conv weights.
                beta, gamma, mean, var = take(cout), take(cout), take(cout), take(cout)
                self._set_folded(name, take(cout, cin, k, k), gamma, beta, mean, var)
        assert pos == len(buf), f"weight file size mismatch: used {pos} of {len(buf)} values"
        return self

    def load(self, path):
        path = str(path)
        return self.load_darknet(path) if path.endswith(".weights") else self.load_checkpoint(path)


def decode(preds, conf_thres=C.CONF_EVAL, iou_thres=C.NMS_IOU, max_det=C.MAX_DET):
    """Raw head outputs -> per-image detections [n,6] = (x1,y1,x2,y2,score,class) in 320x320 pixels.

    This is the part the ARM CPU does on the board: sigmoid/exp box decoding, then per-class NMS.
    """
    from torchvision.ops import batched_nms

    boxes = []
    for p, stride, anchors in zip(preds, C.STRIDES, C.ANCHORS):
        b, _, h, w, _ = p.shape
        gy, gx = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing="ij")
        a = torch.tensor(anchors, device=p.device, dtype=p.dtype).view(1, 3, 1, 1, 2)
        xy = (p[..., 0:2].sigmoid() + torch.stack((gx, gy), -1)) * stride   # box centre
        wh = p[..., 2:4].clamp(-10, 6).exp() * a                            # box size
        score = p[..., 4:5].sigmoid() * p[..., 5:].sigmoid()                # objectness x class
        boxes.append(torch.cat((xy - wh / 2, xy + wh / 2, score), -1).view(b, -1, 4 + p.shape[-1] - 5))
    boxes = torch.cat(boxes, 1)
    out = []
    for x in boxes:
        i, c = (x[:, 4:] > conf_thres).nonzero(as_tuple=True)
        det = torch.cat((x[i, :4], x[i, 4 + c, None], c[:, None].float()), 1)
        keep = batched_nms(det[:, :4], det[:, 4], det[:, 5], iou_thres)[:max_det]
        out.append(det[keep])
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--map", action="store_true", help="run mAP@0.5 on the VOC2007 test set")
    args = ap.parse_args()

    # True IEEE Float32: the GPU's default TF32 mode keeps only ~10 mantissa bits in convolutions.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV3TinyFloat32().load(args.weights).to(device).eval()
    print(f"loaded {args.weights}: {sum(p.numel() for p in model.parameters()):,} parameters, no BatchNorm")

    # Self-check: same output as the training model in model.py (with BatchNorm, not folded).
    from model import YoloV3Tiny
    ref = YoloV3Tiny().to(device).eval()
    if args.weights.endswith(".weights"):
        ref.load_darknet(args.weights)
    else:
        ref.load_state_dict(torch.load(args.weights, map_location=device, weights_only=False)["model"])
    x = torch.rand(2, 3, 320, 320, device=device)
    with torch.no_grad():
        err = max((a - b).abs().max().item() for a, b in zip(model(x), ref(x)))
    print(f"largest difference vs unfolded model.py: {err:.2e}")

    if args.map:
        from evaluate import evaluate, report
        report(evaluate(model, device, split="test"))
