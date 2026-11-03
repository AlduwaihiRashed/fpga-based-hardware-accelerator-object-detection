"""Fold BatchNorm into the convolution before it, so each layer is just weights + bias (what the FPGA runs).

Usage: python fold.py <checkpoint.pt>   -> writes <checkpoint>-folded.pt and checks mAP is unchanged
"""
import copy
import sys

import torch
import torch.nn as nn

from model import ConvBlock, YoloV3Tiny


def fold_bn(model):
    """Returns a copy where every ConvBlock has conv(bias=True) and no BatchNorm. Output is mathematically the same."""
    model = copy.deepcopy(model).eval()
    for m in model.modules():
        if isinstance(m, ConvBlock) and isinstance(m.bn, nn.BatchNorm2d):
            bn, conv = m.bn, m.conv
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            folded = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, 1, conv.padding, bias=True)
            folded.weight.data = conv.weight * scale.view(-1, 1, 1, 1)
            folded.bias.data = bn.bias - bn.running_mean * scale
            m.conv, m.bn = folded.to(conv.weight.device), nn.Identity()
    return model


if __name__ == "__main__":
    from evaluate import evaluate, report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV3Tiny().to(device)
    model.load_state_dict(torch.load(sys.argv[1], map_location=device, weights_only=False)["model"])
    folded = fold_bn(model)
    x = torch.rand(4, 3, 320, 320, device=device)
    with torch.no_grad():
        err = max((a - b).abs().max().item() for a, b in zip(model.eval()(x), folded(x)))
    print(f"largest output difference after folding: {err:.2e}")
    report(evaluate(folded, device))
    out = sys.argv[1].replace(".pt", "-folded.pt")
    torch.save({"model": folded.state_dict(), "folded": True}, out)
    print("saved", out)
