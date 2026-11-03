"""Export the folded Float32 model to ONNX for the ARM baseline (Report 3, Section 4.5), and check it.

The board runs the .onnx file with ONNX Runtime on the ARM CPU. Box decoding + NMS stay outside the
ONNX graph (done in NumPy on the ARM, same as for INT8 and Posit8), so the graph is just the 13 layers.

Checks (on this PC, ONNX Runtime on the CPU):
  1. raw outputs vs PyTorch on real test images
  2. mAP@0.5 on VOC2007 test must be within 0.1 points of the PyTorch Float32 result

Usage: python export_onnx.py /mnt/ssd/Data/senior/runs/float32-v2/best.pt [--skip-map]
Writes <run>/yolov3-tiny-float32.onnx and <run>/yolov3-tiny-float32.json (anchors, classes, thresholds).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import config as C
from yolov3_tiny_float32 import YoloV3TinyFloat32


def export(model, path):
    x = torch.zeros(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    torch.onnx.export(model, (x,), str(path), input_names=["image"], output_names=["head_stride16", "head_stride32"],
                      dynamic_axes={"image": {0: "batch"}, "head_stride16": {0: "batch"}, "head_stride32": {0: "batch"}},
                      opset_version=17, dynamo=False)


def write_metadata(path, weights):
    meta = {
        "weights": str(weights), "input": "RGB, letterbox 320x320 (pad 128, bilinear), float32 = pixel / 255, NCHW",
        "outputs": "raw heads (B,3,H,W,25): tx, ty, tw, th, objectness, 20 class logits; stride 16 then 32",
        "strides": C.STRIDES, "anchors": C.ANCHORS, "leaky_slope": C.LEAKY_SLOPE, "classes": C.CLASSES,
        "conf_eval": C.CONF_EVAL, "conf_live": C.CONF_LIVE, "nms_iou": C.NMS_IOU, "max_det": C.MAX_DET,
    }
    path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--skip-map", action="store_true")
    args = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False  # true IEEE Float32

    model = YoloV3TinyFloat32().load(args.weights).eval()
    out = Path(args.weights).with_name("yolov3-tiny-float32.onnx")
    export(model, out)
    write_metadata(out.with_suffix(".json"), args.weights)
    print("wrote", out)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    run_onnx = lambda imgs: [torch.from_numpy(o).to(imgs.device) for o in sess.run(None, {"image": imgs.cpu().numpy()})]

    # 1. Raw outputs on 16 real test images, both on the CPU.
    from data import VOCDataset
    ds = VOCDataset("test")
    imgs = torch.stack([ds[i][0] for i in range(16)])
    with torch.no_grad():
        err = max(np.abs(a.numpy() - b.numpy()).max() for a, b in zip(model(imgs), run_onnx(imgs)))
    print(f"largest raw output difference, ONNX Runtime vs PyTorch: {err:.2e}")

    # 2. mAP through ONNX Runtime vs PyTorch.
    if not args.skip_map:
        from evaluate import evaluate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m_pt = evaluate(model.to(device), device, split="test")[:, 0].mean() * 100
        m_onnx = evaluate(model, device, split="test", forward=run_onnx)[:, 0].mean() * 100
        ok = abs(m_pt - m_onnx) <= 0.1
        print(f"mAP@0.5 11-point  PyTorch {m_pt:.2f}   ONNX Runtime {m_onnx:.2f}   difference {m_onnx - m_pt:+.2f}  "
              f"-> {'PASS' if ok else 'FAIL'} (limit 0.1)")
