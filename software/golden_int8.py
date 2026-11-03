"""INT8 golden model: the exact integer arithmetic of the accelerator, in NumPy (Report 3.1, 4.1 and 4.2).

All official INT8 mAP numbers come from this file, and the HLS code must match it bit for bit.
It reads the integer parameters saved by yolov3_tiny_int8.py (int8_params.npz).

Per convolution layer (Report 3.1 2.5, 3.3, 3.4):
  1. acc  = sum(x_code * w_code) + b              8x8-bit products, exact 32-bit accumulator (checked)
  2. v    = (acc * m + 2^(n-1)) >> n              requant to Q16.16 in the output's units (M = s_in*s_w/s_out),
                                                  32-bit per-channel multiplier m and shift n, round half up,
                                                  saturated to 32 bits
  3. heads (h15, h22): stop here, v is the Q16.16 logit (real value = v / 65536)
  4. Leaky:  v < 0  ->  (v * 6554 + 2^15) >> 16   (slope 0.1000061, round half up)
  5. code = round(v / 65536), ties to even, clamped to [-128, 127]
Max-pool works on the INT8 codes (layer 11: stride 1, positions outside the map ignored), upsample copies
each code to a 2x2 block, and the concat just stacks codes (both inputs share one scale).

Exactness of step 1: products are at most 128*128 = 2^14 and there are at most 4,608 terms, so every partial
sum is an integer below 2^27. Float64 holds every integer below 2^53 exactly, so a float64 matrix product
gives the exact integer sum in any order (Report 3.1 4.2 allows this "where exact representation is
verified"). The result is still checked to be an integer.

Usage:
    python golden_int8.py /mnt/ssd/Data/senior/runs/int8-ptq --map val      # official INT8 mAP on validation
    python golden_int8.py /mnt/ssd/Data/senior/runs/int8-ptq --check 16     # compare with the fake-quant model
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from yolov3_tiny_float32 import LAYERS

ACC_LIMIT = 2 ** 31          # signed 32-bit accumulator
Q16_MIN, Q16_MAX = -2 ** 31, 2 ** 31 - 1
LEAKY_MUL, LEAKY_SHIFT = 6554, 16


def round_half_even_shift16(v):
    """Q16.16 -> integer, round to nearest, ties to even (int64 in, int64 out)."""
    q = v >> 16
    r = v - (q << 16)
    return q + ((r > 0x8000) | ((r == 0x8000) & (q & 1 == 1)))


def conv(x, w, k):
    """Exact integer convolution, stride 1, zero padding k//2. x: int [cin,H,W], w: int8 [cout,cin,k,k] -> int64."""
    cin, h, wd = x.shape
    xf = x.astype(np.float64)
    if k == 1:
        cols = xf.reshape(cin, h * wd)
    else:
        p = np.pad(xf, ((0, 0), (1, 1), (1, 1)))
        cols = np.stack([p[:, dy:dy + h, dx:dx + wd] for dy in range(3) for dx in range(3)], 1).reshape(cin * 9, h * wd)
    acc = w.reshape(w.shape[0], -1).astype(np.float64) @ cols
    out = np.rint(acc)
    assert np.array_equal(out, acc), "float64 sum was not exact"   # cannot happen within the bound above
    return out.astype(np.int64).reshape(-1, h, wd)


def maxpool(x, stride):
    if stride == 2:
        c, h, w = x.shape
        return x.reshape(c, h // 2, 2, w // 2, 2).max((2, 4))
    p = np.pad(x, ((0, 0), (0, 1), (0, 1)), mode="edge")           # layer 11: outside positions ignored
    return np.maximum.reduce([p[:, :-1, :-1], p[:, 1:, :-1], p[:, :-1, 1:], p[:, 1:, 1:]])


class GoldenInt8:
    def __init__(self, params_path):
        path = Path(params_path)
        if path.is_dir():
            path = path / "int8_params.npz"
        z = np.load(path)
        self.s_x = float(z["s_x"])
        self.p = {name: {key: z[f"{name}.{key}"] for key in ("w", "b", "m", "n")} for name, *_ in LAYERS}
        for name, *_, k, is_head in LAYERS:
            self.p[name]["k"], self.p[name]["head"] = k, is_head
        self.stats = {name: 0 for name, *_ in LAYERS}               # largest |acc| seen per layer

    def input_codes(self, pixels):
        """uint8 RGB [3,H,W] (letterboxed) -> INT8 input codes: round(pixel / 255 / s_x), ties to even.
        On the board the ARM does this step."""
        return np.clip(np.rint(pixels.astype(np.float64) / 255.0 / self.s_x), -128, 127).astype(np.int64)

    def layer(self, name, x, trace=None):
        p = self.p[name]
        if trace is not None:
            trace[name + ".in"] = x
        acc = conv(x, p["w"], p["k"]) + p["b"].astype(np.int64)[:, None, None]
        amax = int(np.abs(acc).max())
        self.stats[name] = max(self.stats[name], amax)
        assert amax < ACC_LIMIT, f"{name}: accumulator overflow ({amax})"
        m, n = p["m"][:, None, None], p["n"][:, None, None]
        v = (acc * m + (np.int64(1) << (n - 1))) >> n               # |acc*m| < 2^31 * 2^31 = 2^62: fits int64
        v = np.clip(v, Q16_MIN, Q16_MAX)
        if trace is not None:
            trace[name + ".acc"] = acc
        if p["head"]:
            return v                                                # Q16.16 logits
        v = np.where(v < 0, (v * LEAKY_MUL + (1 << (LEAKY_SHIFT - 1))) >> LEAKY_SHIFT, v)
        code = np.clip(round_half_even_shift16(v), -128, 127)
        if trace is not None:
            trace[name] = code
        return code

    def forward(self, pixels, trace=None):
        """One letterboxed uint8 RGB image [3,320,320] -> the two heads as int Q16.16 arrays
        [(75,20,20) stride 16, (75,10,10) stride 32]."""
        L = lambda n, t: self.layer(n, t, trace)
        x = maxpool(L("c0", self.input_codes(pixels)), 2)
        x = maxpool(L("c2", x), 2)
        x = maxpool(L("c4", x), 2)
        x = maxpool(L("c6", x), 2)
        r8 = L("c8", x)
        x = maxpool(L("c10", maxpool(r8, 2)), 1)
        r13 = L("c13", L("c12", x))
        out32 = L("h15", L("c14", r13))
        up = L("c18", r13).repeat(2, 1).repeat(2, 2)
        out16 = L("h22", L("c21", np.concatenate((up, r8), 0)))
        return out16, out32


def as_preds(heads, nc=20):
    """Golden head outputs (batch of Q16.16 ints) -> float tensors in the layout decode() expects."""
    import torch
    out = []
    for h in heads:
        t = torch.from_numpy(np.stack(h).astype(np.float64) / 65536.0).float()
        b, _, gh, gw = t.shape
        out.append(t.view(b, 3, 5 + nc, gh, gw).permute(0, 1, 3, 4, 2).contiguous())
    return out


class GoldenForward:
    """Plugs the golden model into evaluate.predict(): takes the float batch (pixel/255), runs every image
    through the integer model on a few CPU threads, returns preds like the PyTorch models."""

    def __init__(self, golden, threads=6):
        self.g, self.pool = golden, ThreadPoolExecutor(threads)

    def eval(self):
        return self

    def __call__(self, imgs):
        pix = np.rint(imgs.cpu().numpy() * 255.0).astype(np.uint8)   # exact: imgs were made as pixel / 255
        res = list(self.pool.map(self.g.forward, pix))
        return [t.to(imgs.device) for t in as_preds(list(zip(*res)))]


if __name__ == "__main__":
    import argparse
    import json
    import time

    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="folder with int8_params.npz (and int8.pt for --check)")
    ap.add_argument("--map", choices=["val", "test"])
    ap.add_argument("--check", type=int, default=0, help="compare with the fake-quant model on N validation images")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    run = Path(args.run)
    g = GoldenInt8(run)

    if args.check:
        from data import VOCDataset
        from yolov3_tiny_int8 import INPUT_OF, load_int8, OUTPUT_OF

        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        fq, _ = load_int8(run / "int8.pt", "cpu")
        fq.eval().double()
        ds = VOCDataset("val")
        names = [n for n, *_ in LAYERS]
        # Each layer is fed the golden model's own input codes, so a difference means the two models do
        # different arithmetic in that layer (not an earlier difference carried forward).
        same = {n: [0, 0, 0] for n in names}       # identical, total, off by more than 1 code
        head_err = 0.0
        t0 = time.time()
        for i in range(args.check):
            img = ds[i][0]
            trace = {}
            heads = g.forward(np.rint(img.numpy() * 255).astype(np.uint8), trace)
            trace["h15"], trace["h22"] = heads[1], heads[0]
            for n in names:
                x = trace[n + ".in"]
                s_in = fq.act_scale(INPUT_OF[n])
                x = torch.from_numpy(x[None]).double() * s_in if n != "c0" else img[None].double()
                with torch.no_grad():
                    y = fq.layer(n, x)[0]
                if n.startswith("h"):
                    ref = torch.round(y * 65536).numpy().astype(np.int64)
                    head_err = max(head_err, np.abs(ref - trace[n]).max() / 65536)
                else:
                    ref = torch.round(y / fq.act_scale(OUTPUT_OF[n])).numpy().astype(np.int64)
                d = np.abs(ref - trace[n])
                same[n][0] += (d == 0).sum()
                same[n][1] += d.size
                same[n][2] += (d > (1 if not n.startswith("h") else 65536)).sum()
        print(f"golden vs fake-quant, layer by layer, on {args.check} validation images ({time.time() - t0:.1f} s):")
        for n, (a, b, c) in same.items():
            unit = "Q16.16 values" if n.startswith("h") else "codes"
            print(f"  {n:4s}  identical {unit} {100 * a / b:8.4f} %   off by more than 1 (code / real unit): {c}")
        print(f"  heads: largest difference {head_err:.6f} (real units)")
        print("largest |accumulator| per layer (limit 2^31 = 2,147,483,648):")
        for n, v in g.stats.items():
            print(f"  {n:4s} {v:>13,d}  ({np.log2(max(v, 1)):.1f} bits)")

    if args.map:
        from evaluate import predict, voc_map, report

        t0 = time.time()
        dets, items = predict(torch.nn.Identity(), torch.device("cpu"), split=args.map, batch=32, workers=4,
                              forward=GoldenForward(g, args.threads))
        aps = voc_map(dets, items)
        report(aps)
        secs = time.time() - t0
        print(f"golden model: {len(items)} images in {secs:.0f} s")
        res = {"split": args.map, "mAP11": float(aps[:, 0].mean() * 100), "mAP_all": float(aps[:, 1].mean() * 100),
               "per_class_AP11": {c: float(a * 100) for c, a in zip(__import__("config").CLASSES, aps[:, 0])},
               "max_abs_acc": {k: int(v) for k, v in g.stats.items()}}
        (run / f"golden_{args.map}.json").write_text(json.dumps(res, indent=1))
