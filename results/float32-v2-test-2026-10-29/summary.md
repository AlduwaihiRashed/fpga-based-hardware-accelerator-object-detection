# Test results: Float32 (folded, PyTorch)

Run on 2026-10-29T12:37:20 with `python test_report.py /mnt/ssd/Data/senior/runs/float32-v2/best.pt`.
Checkpoint `/mnt/ssd/Data/senior/runs/float32-v2/best.pt` (sha256 aa9b742be75affc4...).

## Accuracy (VOC2007 test, 4,952 images, mAP@0.5, conf 0.005, NMS 0.45)

| Model | mAP 11-point (official) | mAP all-point |
|---|---|---|
| Float32 (folded, PyTorch) | 64.58 | 65.93 |

## Per-class AP

| Class | Objects | AP 11-point | AP all-point |
|---|---|---|---|
| aeroplane | 285 | 68.6 | 70.6 |
| bicycle | 337 | 72.8 | 75.9 |
| bird | 459 | 54.1 | 54.5 |
| boat | 263 | 53.2 | 54.3 |
| bottle | 469 | 36.9 | 35.5 |
| bus | 213 | 75.2 | 77.7 |
| car | 1201 | 76.3 | 79.8 |
| cat | 358 | 74.9 | 76.2 |
| chair | 756 | 44.0 | 43.6 |
| cow | 244 | 72.0 | 73.4 |
| diningtable | 206 | 63.8 | 65.3 |
| dog | 489 | 69.2 | 70.4 |
| horse | 348 | 79.1 | 81.8 |
| motorbike | 325 | 76.0 | 78.2 |
| person | 4528 | 71.5 | 73.6 |
| pottedplant | 480 | 37.5 | 35.9 |
| sheep | 242 | 64.6 | 65.3 |
| sofa | 239 | 64.9 | 65.9 |
| train | 282 | 73.8 | 76.2 |
| tvmonitor | 308 | 63.2 | 64.6 |

## Speed on the development PC (Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz; NVIDIA GeForce RTX 5060)

Frame in memory to final boxes, batch 1, conf 0.25, 20 warm-up + 500 timed VOC2007 test images. This is NOT the board result.

| Backend | Preprocess ms | Model ms | Postprocess ms | Total ms (mean / p95) | Sustained FPS |
|---|---|---|---|---|---|
| pytorch-cuda | 0.51 | 4.25 | 1.44 | 6.20 / 7.13 | 161.1 |
| pytorch-cpu | 0.48 | 16.61 | 0.95 | 18.04 / 20.42 | 55.4 |
| onnxruntime-cpu | 0.81 | 10.29 | 1.28 | 12.37 / 14.62 | 80.8 |
