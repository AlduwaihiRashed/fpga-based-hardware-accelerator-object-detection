"""Run the official test of a frozen model and save every record Report 4 needs (Report 3.1, Section 4.6).

One command writes a dated results folder with:
  results.json         every number (mAP, per-class AP, latency, FPS, machine, checkpoint hash)
  summary.md           ready-to-paste tables, generated from results.json (never typed by hand)
  per_class_ap.csv     AP per class, 11-point and all-point
  latency_<backend>.csv  raw per-image timings, one row per image, so anyone can recompute mean/p95
  fig1_per_class_ap.png, fig2_pr_curves.png, fig3_training.png, fig4_latency.png, fig5_detections.png

Accuracy: the folded Float32 reference model on the 4,952 VOC2007 test images, conf 0.005, NMS 0.45,
VOC2007 11-point mAP@0.5 (official) and all-point.
Speed: Report 3.1 rules: frame already in memory -> preprocess -> model -> decode + NMS (conf 0.25)
-> boxes in original pixels. 20 warm-up images, then 500 timed test images, batch 1. Mean and 95th
percentile per stage, and sustained FPS over the whole loop. Measured on THIS PC as a stand-in until the
KV260 arrives; the ONNX Runtime CPU row is the closest stand-in for the ARM baseline.

Usage:
  python test_report.py /mnt/ssd/Data/senior/runs/float32-v2/best.pt
  python test_report.py <best.pt> --skip-speed         # accuracy and figures only
"""
import argparse
import csv
import datetime
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import config as C
from data import letterbox, to_tensor
from evaluate import predict, voc_map
from yolov3_tiny_float32 import YoloV3TinyFloat32, decode

OUT_ROOT = Path(__file__).resolve().parent.parent / "results"

# Colors: validated default categorical palette (dataviz skill, references/palette.md), light mode.
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


# ---------------------------------------------------------------- accuracy

def run_accuracy(model, device):
    dets, items = predict(model, device, split="test")
    aps, pr = voc_map(dets, items, curves=True)
    return dets, items, aps, pr


# ---------------------------------------------------------------- speed

def run_speed(model, backend, images, onnx_path, warmup=20):
    """Returns per-image stage times in ms, and sustained FPS over the timed loop."""
    if backend == "onnxruntime-cpu":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        infer = lambda x: [torch.from_numpy(o) for o in sess.run(None, {"image": x.numpy()})]
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if backend == "pytorch-cuda" else "cpu")
        m = model.to(device).eval()
        infer = lambda x: m(x.to(device))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    rows = []
    with torch.no_grad():
        start = None
        for n, img in enumerate(images):
            if n == warmup:
                sync()
                start = time.perf_counter()
            t0 = time.perf_counter()
            lb, (r, dx, dy) = letterbox(img)                     # preprocess
            x = to_tensor(lb)[None]
            t1 = time.perf_counter()
            preds = infer(x)                                     # model
            sync()
            t2 = time.perf_counter()
            d = decode(preds, conf_thres=C.CONF_LIVE)[0].cpu().numpy()   # postprocess
            d[:, [0, 2]] = (d[:, [0, 2]] - dx) / r
            d[:, [1, 3]] = (d[:, [1, 3]] - dy) / r
            t3 = time.perf_counter()
            if n >= warmup:
                rows.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3))
        total = time.perf_counter() - start
    model.cpu()
    return np.array(rows), len(rows) / total


def stats(a):
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95))}


# ---------------------------------------------------------------- figures

def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    })
    return plt


def fig_per_class(plt, aps, out, label):
    ap = aps[:, 0] * 100
    order = np.argsort(ap)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.arange(20), ap[order], height=0.72, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(np.arange(20), [C.CLASSES[i] for i in order])
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(ap[order]):
        ax.text(v + 0.8, y, f"{v:.1f}", va="center", fontsize=8, color=INK2, zorder=3,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))
    m = ap.mean()
    ax.axvline(m, color=INK, linewidth=1, linestyle="--")
    ax.text(m + 0.8, 19.6, f"mAP {m:.2f}", fontsize=9, color=INK, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_xlabel("AP@0.5 (%), VOC2007 11-point")
    ax.set_title(f"Per-class AP on VOC2007 test: {label}")
    fig.tight_layout()
    fig.savefig(out / "fig1_per_class_ap.png", dpi=200)
    plt.close(fig)


def fig_pr(plt, aps, pr, out, label):
    fig, axs = plt.subplots(4, 5, figsize=(12, 9.5), sharex=True, sharey=True)
    for c, ax in enumerate(axs.flat):
        rec, prec = pr[c]["recall"], pr[c]["precision"]
        ax.plot(rec, prec, color=SERIES[0], linewidth=1.5)
        ax.set_title(f"{C.CLASSES[c]}  AP {aps[c, 0] * 100:.1f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    for ax in axs[-1]:
        ax.set_xlabel("Recall")
    for ax in axs[:, 0]:
        ax.set_ylabel("Precision")
    fig.suptitle(f"Precision-recall curves per class, VOC2007 test: {label}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_pr_curves.png", dpi=200)
    plt.close(fig)


def fig_training(plt, log_path, best_epoch, out):
    ep, loss, vep, vmap = [], [], [], []
    for row in csv.reader(open(log_path)):
        ep.append(int(row[0]))
        loss.append([float(v) for v in row[1:4]])
        if len(row) > 5:
            vep.append(int(row[0]))
            vmap.append(float(row[5]))
    loss = np.array(loss)
    # Two panels, not two y-axes: loss and mAP have different scales.
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for k, name in enumerate(["box (CIoU)", "objectness", "class"]):
        a.plot(ep, loss[:, k], color=SERIES[k], label=name)
        a.text(ep[-1] + 0.8, loss[-1, k], name, color=INK2, fontsize=8, va="center")
    a.set_xlim(0, ep[-1] + 12)
    a.set_xlabel("Epoch")
    a.set_ylabel("Training loss")
    a.set_title("Training loss per part")
    a.legend(frameon=False, loc="upper right")
    b.plot(vep, vmap, color=SERIES[0], marker="o", markersize=4)
    i = vep.index(best_epoch)
    b.plot([best_epoch], [vmap[i]], "o", markersize=9, color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=2)
    b.annotate(f"best: epoch {best_epoch}, {vmap[i]:.2f}", (best_epoch, vmap[i]), xytext=(10, -4),
               textcoords="offset points", fontsize=9, va="top")
    b.set_ylim(top=vmap[i] + 0.4)
    b.set_xlabel("Epoch")
    b.set_ylabel("Validation mAP@0.5 (%), 11-point")
    b.set_title("Validation mAP (model picked here)")
    fig.tight_layout()
    fig.savefig(out / "fig3_training.png", dpi=200)
    plt.close(fig)


def fig_latency(plt, speed, out):
    names = list(speed)
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.7 * len(names)))
    stages = ["preprocess", "model", "postprocess"]
    for y, n in enumerate(names):
        left = 0
        for k, s in enumerate(stages):
            v = speed[n][s]["mean"]
            ax.barh(y, v, left=left, height=0.6, color=SERIES[k], edgecolor=SURFACE, linewidth=2,
                    label=s if y == 0 else None)
            left += v
        ax.text(left + 0.3, y, f"{speed[n]['total']['mean']:.1f} ms mean, {speed[n]['fps']:.0f} FPS sustained",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(speed[n]["total"]["mean"] for n in names) * 1.6)
    ax.set_xlabel("Mean latency per frame (ms), batch 1")
    ax.set_title("Per-stage latency on the development PC (stand-in until the KV260)", pad=26)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig4_latency.png", dpi=200)
    plt.close(fig)


def fig_detections(plt, dets, items, out, n=8, seed=0):
    pick = np.random.default_rng(seed).choice(len(items), n, replace=False)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for ax, i in zip(axs.flat, pick):
        it = items[i]
        ax.imshow(cv2.imread(it["path"])[:, :, ::-1])
        for (x1, y1, x2, y2), c in zip(it["boxes"], it["cls"]):
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=SERIES[1],
                                       linewidth=1.5, linestyle="--"))
        for x1, y1, x2, y2, s, c in dets[i][dets[i][:, 4] >= C.CONF_LIVE]:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=SERIES[0], linewidth=2))
            ax.text(x1, y1 - 3, f"{C.CLASSES[int(c)]} {s:.2f}", fontsize=7, color="white",
                    bbox=dict(facecolor=SERIES[0], edgecolor="none", pad=1))
        ax.set_title(it["id"], fontsize=8, fontweight="normal")
        ax.axis("off")
    fig.suptitle("Sample detections (blue: model, conf >= 0.25; orange dashed: ground truth)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig5_detections.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- records

def write_summary(res, out):
    a, L = res["accuracy"], []
    L += [f"# Test results: {res['label']}", "",
          f"Run on {res['date']} with `{res['command']}`.", f"Checkpoint `{res['weights']}` (sha256 {res['sha256'][:16]}...).", "",
          "## Accuracy (VOC2007 test, 4,952 images, mAP@0.5, conf 0.005, NMS 0.45)", "",
          "| Model | mAP 11-point (official) | mAP all-point |", "|---|---|---|",
          f"| {res['label']} | {a['map11']:.2f} | {a['map']:.2f} |", "",
          "## Per-class AP", "", "| Class | Objects | AP 11-point | AP all-point |", "|---|---|---|---|"]
    for c in a["per_class"]:
        L.append(f"| {c['class']} | {c['objects']} | {c['ap11']:.1f} | {c['ap']:.1f} |")
    if res.get("speed"):
        L += ["", f"## Speed on the development PC ({res['machine']['cpu']}; {res['machine']['gpu']})", "",
              "Frame in memory to final boxes, batch 1, conf 0.25, 20 warm-up + "
              f"{res['speed_images']} timed VOC2007 test images. This is NOT the board result.", "",
              "| Backend | Preprocess ms | Model ms | Postprocess ms | Total ms (mean / p95) | Sustained FPS |",
              "|---|---|---|---|---|---|"]
        for n, s in res["speed"].items():
            L.append(f"| {n} | {s['preprocess']['mean']:.2f} | {s['model']['mean']:.2f} | {s['postprocess']['mean']:.2f} "
                     f"| {s['total']['mean']:.2f} / {s['total']['p95']:.2f} | {s['fps']:.1f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--label", default="Float32 (folded, PyTorch)")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--speed-images", type=int, default=500)
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False  # true IEEE Float32
    weights = Path(args.weights)
    run = weights.parent
    out = OUT_ROOT / f"{run.name}-test-{datetime.date.today()}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV3TinyFloat32().load(weights).to(device).eval()

    print("accuracy on VOC2007 test ...", flush=True)
    dets, items, aps, pr = run_accuracy(model, device)
    per_class = [{"class": C.CLASSES[c], "objects": pr[c]["npos"], "ap11": aps[c, 0] * 100, "ap": aps[c, 1] * 100}
                 for c in range(20)]
    res = {
        "label": args.label, "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": "python test_report.py " + " ".join(f"{k}" for k in __import__("sys").argv[1:]),
        "weights": str(weights), "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "settings": {"img_size": C.IMG_SIZE, "conf_eval": C.CONF_EVAL, "conf_live": C.CONF_LIVE, "nms_iou": C.NMS_IOU,
                     "max_det": C.MAX_DET, "leaky": C.LEAKY_SLOPE, "anchors": C.ANCHORS, "tf32": False},
        "machine": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"},
        "accuracy": {"images": len(items), "objects": sum(p["npos"] for p in pr),
                     "map11": float(aps[:, 0].mean() * 100), "map": float(aps[:, 1].mean() * 100), "per_class": per_class},
    }
    try:  # nicer CPU name on Linux
        res["machine"]["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    print(f"  mAP 11-point {res['accuracy']['map11']:.2f}   all-point {res['accuracy']['map']:.2f}")
    with open(out / "per_class_ap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["class", "objects", "ap11", "ap"])
        w.writeheader()
        w.writerows({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in per_class)

    if not args.skip_speed:
        res["speed_images"] = args.speed_images
        images = [cv2.imread(it["path"]) for it in items[:20 + args.speed_images]]  # frames already in memory
        backends = ["pytorch-cpu"]
        if torch.cuda.is_available():
            backends.insert(0, "pytorch-cuda")
        onnx = run / "yolov3-tiny-float32.onnx"
        if onnx.exists():
            backends.append("onnxruntime-cpu")
        res["speed"] = {}
        for b in backends:
            print(f"speed: {b} ...", flush=True)
            t, fps = run_speed(model, b, images, onnx)
            np.savetxt(out / f"latency_{b}.csv", t, fmt="%.4f", delimiter=",",
                       header="preprocess_ms,model_ms,postprocess_ms,total_ms", comments="")
            res["speed"][b] = {s: stats(t[:, k]) for k, s in enumerate(["preprocess", "model", "postprocess", "total"])}
            res["speed"][b]["fps"] = fps
            print(f"  total {res['speed'][b]['total']['mean']:.2f} ms mean, {fps:.1f} FPS")

    (out / "results.json").write_text(json.dumps(res, indent=1))
    write_summary(res, out)

    plt = style()
    fig_per_class(plt, aps, out, args.label)
    fig_pr(plt, aps, pr, out, args.label)
    if (run / "log.csv").exists() and (run / "final.json").exists():
        fig_training(plt, run / "log.csv", int(json.loads((run / "final.json").read_text())["best_epoch"]), out)
    if res.get("speed"):
        fig_latency(plt, res["speed"], out)
    fig_detections(plt, dets, items, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
