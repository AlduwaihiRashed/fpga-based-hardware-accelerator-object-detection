# Data

Everything the code reads or writes lives here. The code finds this folder by itself; set `SENIOR_DATA` only to use a different folder with the same layout.

| Folder | What | Size | In Git? |
|---|---|---|---|
| `splits/val.txt` | 1,655 VOC07+12 trainval ids held out for validation (seed 0). Used to pick epochs and choose PTQ vs QAT | 17 KB | yes |
| `splits/calib.txt` | 512 training ids (seed 0) used to calibrate INT8 / Posit8 scales | 6 KB | yes |
| `weights/yolov3-tiny.weights` | darknet's COCO-pretrained YOLOv3-Tiny, the starting point for training | 34 MB | yes |
| `runs/float32-v2/` | **Official Float32 model**: `best.pt` (epoch 28), training log, `final.json`, ONNX export for the ARM | 134 MB | yes, except `last.pt` |
| `runs/int8-ptq/` | INT8 PTQ: `int8.pt` (fake quant), `int8_params.npz` (all integers, for the hardware), `calibration.json`, `golden_val.json` | 42 MB | yes |
| `runs/float32-v1/` | First Float32 run (64.09 mAP), obsolete, kept for the record | 100 MB | no |
| `datasets/VOCdevkit/` | PASCAL VOC 2007 + 2012, unpacked (what the code reads) | 2.9 GB | no |
| `datasets/*.tar` | The original VOC downloads (same content as `VOCdevkit/`) | 2.8 GB | no |

Getting the dataset without the shared folder: download `VOCtrainval_06-Nov-2007.tar`, `VOCtest_06-Nov-2007.tar` and `VOCtrainval_11-May-2012.tar` from the PASCAL VOC site (or a mirror), put them in `datasets/`, and unpack them there with `tar xf <file>`. They unpack into `datasets/VOCdevkit/`.

`datasets/VOCdevkit/cache_*.pkl` files are rebuilt automatically on first use and should never be copied between machines.
