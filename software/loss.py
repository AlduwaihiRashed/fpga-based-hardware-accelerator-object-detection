"""YOLOv3 training loss: CIoU box loss + objectness BCE + class BCE."""
import math

import torch
import torch.nn.functional as F

import config as C

BOX_GAIN, OBJ_GAIN, CLS_GAIN = 0.075, 0.4, 0.19
OBJ_BALANCE = (4.0, 1.0)   # per head, same order as config.STRIDES
ANCHOR_RATIO = 4.0         # a target uses every anchor whose w and h are within 4x of its own


def ciou(b1, b2, eps=1e-7):
    """Complete IoU of matched boxes given as (cx, cy, w, h)."""
    x1, y1, x2, y2 = b1[:, 0] - b1[:, 2] / 2, b1[:, 1] - b1[:, 3] / 2, b1[:, 0] + b1[:, 2] / 2, b1[:, 1] + b1[:, 3] / 2
    X1, Y1, X2, Y2 = b2[:, 0] - b2[:, 2] / 2, b2[:, 1] - b2[:, 3] / 2, b2[:, 0] + b2[:, 2] / 2, b2[:, 1] + b2[:, 3] / 2
    inter = (torch.min(x2, X2) - torch.max(x1, X1)).clamp(0) * (torch.min(y2, Y2) - torch.max(y1, Y1)).clamp(0)
    iou = inter / (b1[:, 2] * b1[:, 3] + b2[:, 2] * b2[:, 3] - inter + eps)
    c2 = (torch.max(x2, X2) - torch.min(x1, X1)) ** 2 + (torch.max(y2, Y2) - torch.min(y1, Y1)) ** 2 + eps
    rho2 = (b1[:, 0] - b2[:, 0]) ** 2 + (b1[:, 1] - b2[:, 1]) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(b2[:, 2] / (b2[:, 3] + eps)) - torch.atan(b1[:, 2] / (b1[:, 3] + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - rho2 / c2 - alpha * v


def compute_loss(preds, targets):
    """preds: model output list; targets: [m,6] = (image index, class, cx, cy, w, h) normalised."""
    dev = targets.device
    lbox = lobj = lcls = torch.zeros((), device=dev)
    for p, stride, anchors, balance in zip(preds, C.STRIDES, C.ANCHORS, OBJ_BALANCE):
        b, _, h, w, _ = p.shape
        a = torch.tensor(anchors, device=dev) / stride           # anchors in grid cells
        tobj = torch.zeros((b, 3, h, w), device=dev)
        if len(targets):
            gxy = targets[:, 2:4] * torch.tensor([w, h], device=dev)
            gwh = targets[:, 4:6] * torch.tensor([w, h], device=dev)
            r = gwh[:, None] / a[None]
            ti, ai = (torch.max(r, 1 / r).amax(2) < ANCHOR_RATIO).nonzero(as_tuple=True)
            if len(ti):
                bi, ci = targets[ti, 0].long(), targets[ti, 1].long()
                gi = gxy[ti, 0].long().clamp(0, w - 1)
                gj = gxy[ti, 1].long().clamp(0, h - 1)
                ps = p[bi, ai, gj, gi]
                pxy = ps[:, :2].sigmoid() + torch.stack((gi, gj), 1)
                pwh = ps[:, 2:4].clamp(-10, 6).exp() * a[ai]
                iou = ciou(torch.cat((pxy, pwh), 1), torch.cat((gxy[ti], gwh[ti]), 1))
                lbox = lbox + (1 - iou).mean()
                tobj[bi, ai, gj, gi] = iou.detach().clamp(0)
                tcls = F.one_hot(ci, p.shape[-1] - 5).float()
                lcls = lcls + F.binary_cross_entropy_with_logits(ps[:, 5:], tcls)
        lobj = lobj + balance * F.binary_cross_entropy_with_logits(p[..., 4], tobj)
    parts = torch.stack((BOX_GAIN * lbox, OBJ_GAIN * lobj, CLS_GAIN * lcls))
    return parts.sum() * preds[0].shape[0], parts.detach()
