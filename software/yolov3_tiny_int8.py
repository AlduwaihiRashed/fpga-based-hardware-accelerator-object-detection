"""YOLOv3-Tiny INT8: fake quantization in PyTorch, MSE calibration and post-training quantization (PTQ).

Follows Report 3.1 (sections 2.5, 3.3, 3.4, 4.1):
  - symmetric INT8, no zero point, codes -128..127, round to nearest with ties to even
  - weights: one scale per output channel; activations: one scale per tensor
  - every scale picked by the lowest mean squared error (MSE) search, on the 512 calibration images
  - input = pixel / 255, quantized with its own scale
  - bias = INT32 at scale s_in * s_w, added in the 32-bit accumulator
  - the two tensors joined by the concat (upsampled c18 and c8) share one scale
  - the two heads skip Leaky and the INT8 encoder; they come out as Q16.16 numbers

This file only *simulates* INT8 with float numbers ("fake quantization"), which is fast and lets us train (QAT).
The exact integer arithmetic of the FPGA lives in golden_int8.py, and all official INT8 mAP numbers come from
there. int_params() below turns this model into the integers the golden model and the board use.

Usage:
    python yolov3_tiny_int8.py /mnt/ssd/Data/senior/runs/float32-v2/best.pt --name int8-ptq           # calibrate + save
    python yolov3_tiny_int8.py /mnt/ssd/Data/senior/runs/int8-ptq/int8.pt --map val                   # mAP of a saved model
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as C
from yolov3_tiny_float32 import LAYERS, YoloV3TinyFloat32

QMIN, QMAX = -128, 127
Q16 = 1 << 16             # Q16.16: 16 fraction bits

# Which activation tensor feeds each layer. "x" is the input image, "cat" is the concat tensor,
# whose scale is shared by the c8 and c18 outputs (Report 3.1 2.5).
INPUT_OF = {"c0": "x", "c2": "c0", "c4": "c2", "c6": "c4", "c8": "c6", "c10": "cat", "c12": "c10",
            "c13": "c12", "c14": "c13", "h15": "c14", "c18": "c13", "c21": "cat", "h22": "c21"}
# Which scale each (non-head) layer's output is encoded with.
OUTPUT_OF = {name: ("cat" if name in ("c8", "c18") else name) for name, *_ in LAYERS if not name.startswith("h")}
ACTS = ["x", "c0", "c2", "c4", "c6", "cat", "c10", "c12", "c13", "c14", "c21"]


def fake_quant(x, s):
    """Quantize to INT8 at scale s and back to float (s: number or tensor that broadcasts).
    Ties go to even (torch.round). Gradient: straight through inside the range, zero where clipped (for QAT)."""
    xc = torch.maximum(torch.minimum(x, QMAX * s), QMIN * s)
    return xc + (torch.round(xc / s) * s - xc).detach()


def mse_scale(values, counts, amax, n_cand=400, lo=0.02):
    """Lowest-MSE symmetric INT8 scale for a histogram (bin centres `values`, `counts`).
    Candidates: s = a * amax / 127 for a from `lo` to 1 (a < 1 clips the largest values)."""
    a = torch.linspace(lo, 1.0, n_cand, device=values.device, dtype=torch.float64)
    s = (a * amax / QMAX)[:, None]
    q = torch.clamp(torch.round(values[None] / s), QMIN, QMAX) * s
    err = ((values[None] - q) ** 2 * counts[None]).sum(1)
    return float(s[err.argmin(), 0])


def weight_scales(w, n_cand=200):
    """Per-output-channel lowest-MSE scale for a conv weight [cout, cin, k, k]. Returns [cout]."""
    w = w.detach().double().flatten(1)
    amax = w.abs().amax(1).clamp_min(1e-12)
    a = torch.linspace(0.3, 1.0, n_cand, device=w.device, dtype=torch.float64)
    best_s, best_err = amax / QMAX, torch.full_like(amax, float("inf"))
    for ai in a:                                       # one candidate at a time keeps memory small
        s = (ai * amax / QMAX)[:, None]
        err = ((w - torch.clamp(torch.round(w / s), QMIN, QMAX) * s) ** 2).sum(1)
        better = err < best_err
        best_s, best_err = torch.where(better, s[:, 0], best_s), torch.where(better, err, best_err)
    return best_s.float()


class YoloV3TinyInt8(YoloV3TinyFloat32):
    """Same network as the Float32 model; only layer() changes. Scales are buffers so they save with the model."""

    def __init__(self, num_classes=C.NUM_CLASSES):
        super().__init__(num_classes)
        self.mode = "float"                     # "float" | "record" (calibration) | "int8"
        for a in ACTS:
            self.register_buffer(f"s_{a}", torch.tensor(1.0))
        for name, cin, cout, k, is_head in LAYERS:
            self.register_buffer(f"sw_{name}", torch.ones(self.conv[name].out_channels))
        self._rec = None

    def act_scale(self, a):
        return getattr(self, f"s_{a}")

    def layer(self, name, x):
        if self.mode == "float":
            return super().layer(name, x)
        if self.mode == "record":
            if name == "c0":
                self._record("x", x)
            y = super().layer(name, x)
            if not name.startswith("h"):
                # the upsample copies every c18 value 4 times into the concat tensor, so weight it 4x
                self._record(OUTPUT_OF[name], y, weight=4.0 if name == "c18" else 1.0)
            return y

        # ---- INT8 (fake quantization) ----
        conv = self.conv[name]
        s_in = self.act_scale(INPUT_OF[name])
        s_w = getattr(self, f"sw_{name}")
        x = fake_quant(x, s_in)                                   # no-op except for the input image
        w = fake_quant(conv.weight, s_w.view(-1, 1, 1, 1))
        s_b = s_in * s_w                                          # INT32 bias scale
        b = conv.bias + (torch.round(conv.bias / s_b) * s_b - conv.bias).detach()
        y = F.conv2d(x, w, b, padding=conv.padding)
        if name.startswith("h"):
            return y + (torch.round(y * Q16) / Q16 - y).detach()  # Q16.16 head output
        y = F.leaky_relu(y, C.LEAKY_SLOPE)
        return fake_quant(y, self.act_scale(OUTPUT_OF[name]))

    # ---------------- calibration ----------------

    def _record(self, a, t, weight=1.0):
        """Two passes: first the largest |value| per tensor, then a histogram over [-amax, amax]."""
        t = t.detach().float()
        r = self._rec
        if r["pass"] == 0:
            r["amax"][a] = max(r["amax"].get(a, 0.0), t.abs().max().item())
        else:
            m = r["amax"][a]
            h = torch.histc(t, bins=r["bins"], min=-m, max=m).double() * weight
            r["hist"][a] = r["hist"].get(a, 0) + h

    @torch.no_grad()
    def calibrate(self, loader, device, bins=8192):
        """Pick all scales by lowest MSE (Report 3.1 4.1 step 4). Activations are measured on the Float32 model."""
        for name, *_ in LAYERS:
            getattr(self, f"sw_{name}").copy_(weight_scales(self.conv[name].weight))
        self.mode, self._rec = "record", {"pass": 0, "amax": {}, "hist": {}, "bins": bins}
        for p in (0, 1):
            self._rec["pass"] = p
            for imgs, *_ in loader:
                self(imgs.to(device))
        for a in ACTS:
            m = self._rec["amax"][a]
            edges = torch.linspace(-m, m, bins + 1, device=device, dtype=torch.float64)
            centres = (edges[1:] + edges[:-1]) / 2
            self.act_scale(a).fill_(mse_scale(centres, self._rec["hist"][a], m))
        info = {a: {"amax": self._rec["amax"][a], "scale": self.act_scale(a).item(),
                    "clip_at": QMAX * self.act_scale(a).item()} for a in ACTS}
        self._rec, self.mode = None, "int8"
        return info

    # ---------------- integer parameters (for golden_int8.py, export and the HLS code) ----------------

    @torch.no_grad()
    def int_params(self):
        """Everything the accelerator needs, as integers.

        Per layer:  w  int8 [cout, cin, k, k]   weight codes
                    b  int32 [cout]             bias at scale s_in * s_w
                    m  int32 [cout], n int [cout]   requant: Q16.16 output = (acc * m + 2^(n-1)) >> n
                                                 which equals acc * M * 2^16 with M = s_in * s_w / s_out
                                                 (s_out = 1 for the heads, so they come out as the real value)
        Plus the input scale s_x: input code = round(pixel / 255 / s_x), ties to even.
        """
        out = {"s_x": self.s_x.item(), "layers": {}}
        for name, cin, cout, k, is_head in LAYERS:
            conv = self.conv[name]
            s_in = self.act_scale(INPUT_OF[name]).double()
            s_w = getattr(self, f"sw_{name}").double()
            s_out = 1.0 if is_head else self.act_scale(OUTPUT_OF[name]).double()
            w = torch.clamp(torch.round(conv.weight.double() / s_w.view(-1, 1, 1, 1)), QMIN, QMAX)
            b = torch.round(conv.bias.double() / (s_in * s_w))
            assert b.abs().max() < 2 ** 31, f"{name}: bias does not fit INT32"
            m, n = multiplier(s_in * s_w / s_out * Q16)
            out["layers"][name] = {
                "w": w.cpu().numpy().astype(np.int8), "b": b.cpu().numpy().astype(np.int32),
                "m": m, "n": n, "head": is_head, "k": k,
                "s_in": float(s_in), "s_w": s_w.cpu().numpy().astype(np.float32), "s_out": float(s_out),
            }
        return out


def multiplier(M16):
    """Real multipliers (per channel) -> 32-bit integer m in [2^30, 2^31) and right shift n, so m / 2^n ~= M16."""
    m, n = [], []
    for v in M16.tolist():
        e = math.floor(math.log2(v))
        sh = 30 - e
        mi = round(v * 2 ** sh)
        if mi == 2 ** 31:               # rounding pushed it over: one bit less
            mi, sh = mi // 2, sh - 1
        assert 1 <= sh <= 62, f"requant shift {sh} out of range"
        m.append(mi)
        n.append(sh)
    return np.array(m, np.int64), np.array(n, np.int64)


def load_int8(path, device="cpu"):
    """Load a model saved by this script (int8.pt)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = YoloV3TinyInt8()
    model.load_state_dict(ck["model"])
    model.mode = "int8"
    return model.to(device), ck


def save_params(params, path):
    """Integer parameters as one .npz (read by golden_int8.py)."""
    flat = {"s_x": np.float64(params["s_x"])}
    for name, p in params["layers"].items():
        for key in ("w", "b", "m", "n", "s_w"):
            flat[f"{name}.{key}"] = p[key]
        flat[f"{name}.s_in"], flat[f"{name}.s_out"] = np.float64(p["s_in"]), np.float64(p["s_out"])
    np.savez(path, **flat)


if __name__ == "__main__":
    import argparse
    from data import VOCDataset, collate
    from evaluate import evaluate, report

    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="Float32 checkpoint (best.pt) to calibrate, or a saved int8.pt")
    ap.add_argument("--name", default="int8-ptq", help="output folder under runs/")
    ap.add_argument("--map", choices=["val", "test"], help="also measure fake-quant mAP on this split")
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Path(args.weights).name.startswith("int8"):
        model, _ = load_int8(args.weights, device)
    else:
        model = YoloV3TinyInt8()
        model.load(args.weights)
        model.to(device).eval()
        calib = DataLoader(VOCDataset("calib"), 32, num_workers=8, collate_fn=collate)
        info = model.calibrate(calib, device)
        out = C.RUNS_DIR / args.name
        out.mkdir(parents=True, exist_ok=True)
        params = model.int_params()
        torch.save({"model": model.state_dict(), "source": str(args.weights), "calibration": info}, out / "int8.pt")
        save_params(params, out / "int8_params.npz")
        summary = {
            "source": str(args.weights), "calib_images": C.CALIB_IMAGES, "calib_seed": C.CALIB_SEED,
            "activations": info,
            "weights": {n: {"scale_min": float(p["s_w"].min()), "scale_max": float(p["s_w"].max()),
                            "requant_shift_min": int(p["n"].min()), "requant_shift_max": int(p["n"].max())}
                        for n, p in params["layers"].items()},
        }
        (out / "calibration.json").write_text(json.dumps(summary, indent=1))
        print(f"saved {out}/int8.pt, int8_params.npz, calibration.json")
        for a, v in info.items():
            print(f"  {a:4s} max {v['amax']:8.3f}   scale {v['scale']:.5f}   clips above {v['clip_at']:7.3f}")

    if args.map:
        model.eval()
        report(evaluate(model, device, split=args.map))
