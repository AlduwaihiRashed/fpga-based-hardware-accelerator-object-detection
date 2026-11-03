"""Shared settings for the YOLOv3-Tiny software work (Float32, INT8, Posit8)."""
import os
from pathlib import Path

# Datasets, weights, training runs and splits live in the repo's data/ folder.
# Set SENIOR_DATA to use a different folder with the same layout (for example on Colab).
DATA_ROOT = Path(os.environ.get("SENIOR_DATA", Path(__file__).resolve().parent.parent / "data"))
VOC_ROOT = DATA_ROOT / "datasets" / "VOCdevkit"
WEIGHTS_DIR = DATA_ROOT / "weights"
RUNS_DIR = DATA_ROOT / "runs"

IMG_SIZE = 320          # DC4: fixed 320x320 letterbox input
PAD_VALUE = 128         # letterbox padding colour (grey)
NUM_CLASSES = 20        # DC5: PASCAL VOC
CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# Anchors (w, h) in pixels at 320x320, from k-means (1 - IoU distance) on the training boxes (Report 3, 4.1 step 1).
# Recompute with: python anchors.py. Smallest three go to the fine grid, largest three to the coarse grid.
# Mean best IoU with the training boxes: 0.637 (darknet's released anchors: 0.583).
_ANCHORS = [(19, 24), (35, 52), (69, 83), (100, 166), (170, 100), (232, 204)]
# One entry per detection head, in the order the model returns them.
STRIDES = (16, 32)                       # 20x20 grid, then 10x10 grid
ANCHORS = (_ANCHORS[:3], _ANCHORS[3:])

# Index of each VOC class in the 80-class COCO list, used to carry over the pretrained head weights.
COCO_INDEX = (4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62)

# Leaky ReLU slope (Report 3). The hardware approximates it as x * 6554 >> 16 = 0.1000061.
LEAKY_SLOPE = 0.1

# Evaluation protocol (FR6). 0.45 is the usual VOC value; Report 2 left the IoU value blank.
CONF_EVAL = 0.005
CONF_LIVE = 0.25
NMS_IOU = 0.45
MAX_DET = 200

# INT8 / Posit8 calibration (Report 3.1, 4.1 step 3): 512 images from the training split, fixed seed.
CALIB_IMAGES = 512
CALIB_SEED = 0
