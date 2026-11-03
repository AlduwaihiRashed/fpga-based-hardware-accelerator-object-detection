"""YOLOv3-Tiny, layer-for-layer the same as darknet's yolov3-tiny.cfg."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class ConvBlock(nn.Module):
    """Conv + BatchNorm + LeakyReLU (darknet 'convolutional' with batch_normalize=1)."""

    def __init__(self, cin, cout, k, slope=C.LEAKY_SLOPE):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, 1, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MaxPoolStride1(nn.Module):
    """Darknet's size-2 stride-1 maxpool: output keeps the input size (pads right and bottom)."""

    def forward(self, x):
        return F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode="replicate"), 2, 1)


class YoloV3Tiny(nn.Module):
    # Comments give the darknet layer index and the output shape for a 320x320 input.
    def __init__(self, num_classes=C.NUM_CLASSES):
        super().__init__()
        self.nc = num_classes
        no = 3 * (5 + num_classes)
        self.c0 = ConvBlock(3, 16, 3)        # 0   16 x 320 x 320
        self.c2 = ConvBlock(16, 32, 3)       # 2   32 x 160 x 160
        self.c4 = ConvBlock(32, 64, 3)       # 4   64 x 80 x 80
        self.c6 = ConvBlock(64, 128, 3)      # 6   128 x 40 x 40
        self.c8 = ConvBlock(128, 256, 3)     # 8   256 x 20 x 20   (route to 20)
        self.c10 = ConvBlock(256, 512, 3)    # 10  512 x 10 x 10
        self.p11 = MaxPoolStride1()          # 11  512 x 10 x 10
        self.c12 = ConvBlock(512, 1024, 3)   # 12  1024 x 10 x 10
        self.c13 = ConvBlock(1024, 256, 1)   # 13  256 x 10 x 10   (route to 17)
        self.c14 = ConvBlock(256, 512, 3)    # 14  512 x 10 x 10
        self.h15 = nn.Conv2d(512, no, 1)     # 15  75 x 10 x 10    (head, stride 32)
        self.c18 = ConvBlock(256, 128, 1)    # 18  128 x 10 x 10
        self.c21 = ConvBlock(384, 256, 3)    # 21  256 x 20 x 20   (input = upsample(18) ++ 8)
        self.h22 = nn.Conv2d(256, no, 1)     # 22  75 x 20 x 20    (head, stride 16)
        self._init_heads()

    def _init_heads(self):
        # Start with low objectness so early training is stable.
        for head, stride in ((self.h22, 16), (self.h15, 32)):
            b = head.bias.data.view(3, -1)
            b[:, 4] = np.log(8 / (C.IMG_SIZE / stride) ** 2)
            b[:, 5:] = np.log(0.6 / (self.nc - 0.99))

    def forward(self, x):
        """Returns raw head outputs [(B,3,20,20,5+nc), (B,3,10,10,5+nc)] in the order of config.STRIDES."""
        x = F.max_pool2d(self.c0(x), 2)
        x = F.max_pool2d(self.c2(x), 2)
        x = F.max_pool2d(self.c4(x), 2)
        x = F.max_pool2d(self.c6(x), 2)
        r8 = self.c8(x)
        x = self.p11(self.c10(F.max_pool2d(r8, 2)))
        r13 = self.c13(self.c12(x))
        out32 = self.h15(self.c14(r13))
        x = F.interpolate(self.c18(r13), scale_factor=2, mode="nearest")
        out16 = self.h22(self.c21(torch.cat((x, r8), 1)))
        return [self._reshape(out16), self._reshape(out32)]

    def _reshape(self, o):
        b, _, h, w = o.shape
        return o.view(b, 3, 5 + self.nc, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def load_darknet(self, path, class_index=C.COCO_INDEX):
        """Load darknet's COCO yolov3-tiny.weights, keeping only our classes' channels in the two heads."""
        with open(path, "rb") as f:
            major, minor, _ = np.fromfile(f, np.int32, 3)
            np.fromfile(f, np.int64 if major * 10 + minor >= 2 else np.int32, 1)
            buf = np.fromfile(f, np.float32)
        pos = 0

        def take(shape):
            nonlocal pos
            n = int(np.prod(shape))
            t = torch.from_numpy(buf[pos:pos + n].copy()).view(*shape)
            pos += n
            return t

        def block(m):
            for t in (m.bn.bias, m.bn.weight, m.bn.running_mean, m.bn.running_var, m.conv.weight):
                t.data.copy_(take(t.shape))

        def head(m):
            no = 5 + 80  # values per anchor in the COCO head
            keep = [a * no + k for a in range(3) for k in [0, 1, 2, 3, 4] + [5 + c for c in class_index]]
            m.bias.data.copy_(take((3 * no,))[keep])
            m.weight.data.copy_(take((3 * no, m.in_channels, 1, 1))[keep])

        for m in (self.c0, self.c2, self.c4, self.c6, self.c8, self.c10, self.c12, self.c13, self.c14):
            block(m)
        head(self.h15)
        block(self.c18)
        block(self.c21)
        head(self.h22)
        assert pos == len(buf), f"weight file size mismatch: used {pos} of {len(buf)} values"

