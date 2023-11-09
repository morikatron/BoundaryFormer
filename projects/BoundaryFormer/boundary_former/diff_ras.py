from random import random
from typing import List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks
from detectron2.utils.events import get_event_storage

from boundary_former.layers.diff_ras.polygon import SoftPolygon
from boundary_former.utils import get_union_box, rasterize_instances, POLY_LOSS_REGISTRY

class ClippingStrategy(nn.Module):
    def __init__(self, cfg, is_boundary=False):
        super().__init__()

        self.register_buffer("laplacian", torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3))

        self.is_boundary = is_boundary
        self.side_lengths = np.array(cfg.MODEL.DIFFRAS.RESOLUTIONS).reshape(-1, 2)

    # not used.
    def _extract_target_boundary(self, masks, shape):
        boundary_targets = F.conv2d(masks.unsqueeze(1), self.laplacian, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        # odd? only if the width doesn't match?
        if boundary_targets.shape[-2:] != shape:
            boundary_targets = F.interpolate(
                boundary_targets, shape, mode='nearest')

        return boundary_targets

    def forward(self, instances, clip_boxes=None, lid=0):                
        device = self.laplacian.device

        gt_masks = []

        if clip_boxes is not None:
            clip_boxes = torch.split(clip_boxes, [len(inst) for inst in instances], dim=0)
            
        for idx, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue

            if clip_boxes is not None:
                # todo, need to support rectangular boxes.
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    clip_boxes[idx].detach(), self.side_lengths[lid][0])
            else:
                gt_masks_per_image = instances_per_image.gt_masks.rasterize_no_crop(self.side_length).to(device)

            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        return torch.cat(gt_masks).squeeze(1)

def dice_loss(input, target, weight_FP=1, weight_FN=1):
    smooth = 1.

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    # return 1 - ((2. * intersection + smooth) /
    #           (iflat.sum() + tflat.sum() + smooth))

    # NOTE SEN
    # the original equals to (iflat.sum() - intersection + tflat.sum() - intersection) / (iflat.sum() + tflat.sum() + smooth)
    # an approximate is
    # ((iflat-iflat*tflat).sum() + (tflat-iflat*tflat).sum()) / (iflat.sum() + tflat.sum() + smooth)
    # left term is False-Positive, right term is False-Negative
    return ((iflat-iflat*tflat).sum()*weight_FP + (tflat-iflat*tflat).sum()*weight_FN) / (iflat.sum() + tflat.sum() + smooth)


@POLY_LOSS_REGISTRY.register()
class MaskRasterizationLoss(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.register_buffer("rasterize_at", torch.from_numpy(np.array(cfg.MODEL.DIFFRAS.RESOLUTIONS).reshape(-1, 2)))
        self.inv_smoothness_schedule = cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_SCHED
        self.inv_smoothness = self.inv_smoothness_schedule[0]
        self.inv_smoothness_iter = cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_STEPS
        self.inv_smoothness_idx = 0
        self.iter = 0

        # whether to invoke our own rasterizer in "hard" mode.
        self.use_rasterized_gt = cfg.MODEL.DIFFRAS.USE_RASTERIZED_GT
        
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        self.clip_to_proposal = not cfg.MODEL.ROI_HEADS.PROPOSAL_ONLY_GT
        self.predict_in_box_space = cfg.MODEL.BOUNDARY_HEAD.PRED_WITHIN_BOX
        
        if self.clip_to_proposal or not self.use_rasterized_gt:
            self.clipper = ClippingStrategy(cfg)
            self.gt_rasterizer = None
        else:
            self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="hard_mask")

        self.offset = 0.5
        self.loss_fn = lambda i, t: dice_loss(i, t, cfg.MODEL.DIFFRAS.DICE_LOSS_WEIGHT_FP, cfg.MODEL.DIFFRAS.DICE_LOSS_WEIGHT_FN)
        self.name = "mask"

    def _create_targets(self, instances, clip_boxes=None, lid=0):
        if self.clip_to_proposal or not self.use_rasterized_gt:
            targets = self.clipper(instances, clip_boxes=clip_boxes, lid=lid)            
        else:            
            targets = rasterize_instances(self.gt_rasterizer, instances, self.rasterize_at)
            
        return targets

    def forward(self, preds, targets, lid=0):
        if isinstance(targets, torch.Tensor):
            device = targets.device
        else:
            device = targets[0].gt_boxes.device
            
        batch_size = len(preds)
        storage = get_event_storage()

        resolution = self.rasterize_at[lid]

        # advance to the right schedule.
        # this is really bad, please clean up.
        if (self.iter == 0) and (storage.iter > 0):
            for idx, start_iter in enumerate(self.inv_smoothness_iter):
                if start_iter > storage.iter:
                    self.inv_smoothness_idx = idx # - 1???
                    self.inv_smoothness = self.inv_smoothness_schedule[self.inv_smoothness_idx]
                    self.pred_rasterizer.inv_smoothness = self.inv_smoothness
                    break

            if len(self.inv_smoothness_iter) and (storage.iter > self.inv_smoothness_iter[-1]):
                self.inv_smoothness_idx = len(self.inv_smoothness_iter)
                self.inv_smoothness = self.inv_smoothness_schedule[self.inv_smoothness_idx]
                self.pred_rasterizer.inv_smoothness = self.inv_smoothness

            self.iter = 1
        else:
            self.iter = 1

        if ((len(self.inv_smoothness_schedule) > (self.inv_smoothness_idx + 1)) and
            (storage.iter == self.inv_smoothness_iter[self.inv_smoothness_idx])):
            self.inv_smoothness = self.inv_smoothness_schedule[self.inv_smoothness_idx + 1]
            self.pred_rasterizer.inv_smoothness = self.inv_smoothness
            self.inv_smoothness_idx += 1
            self.iter = 1

        # -0.5 needed to align the rasterizer with COCO.
        pred_masks = self.pred_rasterizer(preds * float(resolution[1].item()) - self.offset, resolution[1].item(), resolution[0].item(), 1.0).unsqueeze(1)
        # add a little noise since depending on inv_smoothness/predictions, we can exactly predict 0 or 1.0        
        pred_masks = torch.clamp(pred_masks, 0.00001, 0.99999)

        # NOTE: this is only compatible when pred_box is not changed.
        
        if isinstance(targets, torch.Tensor):            
            target_masks = targets            
        else:
            # if not pooled, we can:
            # enforce a full image loss (unlikely to be efficient)
            # rasterize only within the union of the GT box and the predicted polygon.
            # rasterize only within the predicted polygon (like an RoI) and clip the GT to that.
            clip_boxes = None
            if self.predict_in_box_space:
                # keep preds the same, but rasterize proposal box as GT.
                clip_boxes = torch.cat([t.proposal_boxes.tensor for t in targets])
            
            target_masks = self._create_targets(targets, clip_boxes=clip_boxes, lid=lid)
            target_masks = target_masks.unsqueeze(1).float()
            
        return self.loss_fn(pred_masks, target_masks), target_masks


@POLY_LOSS_REGISTRY.register()
class PolySmoothnessLoss(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.wtype = cfg.MODEL.DIFFRAS.POLY_SMOOTH_LOSS_WEIGHT_TYPE
        self.ws = cfg.MODEL.DIFFRAS.POLY_SMOOTH_LOSS_WEIGHT_POINTWISE

        self.name = 'polysmooth'

    def forward(self, verts, targets, lid=0):
        """verts: [b*n, pts, 2]"""
        device = targets.device if isinstance(targets, torch.Tensor) else targets[0].gt_boxes.device

        edges = torch.roll(verts, 1, 1) - verts
        edges = torch.nn.functional.normalize(edges, dim=2)
        edges_ = torch.roll(edges, 1, 1)

        edges = torch.reshape(edges, [-1, 2])
        edges_ = torch.reshape(edges_, [-1, 2])
        cos = torch.sum(edges * edges_, dim=1)
        loss = ((1-cos)/2) ** 2
        loss = torch.reshape(loss, [verts.shape[0], verts.shape[1]])  # [b*n, pts]

        # weighted
        sorted_loss, _ = torch.sort(loss, descending=True)

        if self.wtype == 'linear':
            weights = torch.clip(torch.arange(verts.shape[1], device=device) - 3, 0) * 0.1
        else:
            weights = torch.ones([verts.shape[1]], device=device)
            for i, w in enumerate(self.ws):
                if i > weights.shape[0]-1:
                    break
                weights[i] = w
        loss = sorted_loss * weights

        loss = torch.sum(loss)
        loss /= verts.shape[0]
        loss /= 4  # NOTE 4 cuz' usually rect. verts.shape[1]

        return loss, targets


from .transformer import MLP
import math
@POLY_LOSS_REGISTRY.register()
class PolyContrastiveLoss(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.wtype = cfg.MODEL.DIFFRAS.POLY_SMOOTH_LOSS_WEIGHT_TYPE
        self.ws = cfg.MODEL.DIFFRAS.POLY_SMOOTH_LOSS_WEIGHT_POINTWISE

        if cfg.MODEL.BOUNDARY_HEAD.UPSAMPLING:
            self.last_lid = int(math.log2(cfg.MODEL.BOUNDARY_HEAD.POLY_NUM_PTS // cfg.MODEL.BOUNDARY_HEAD.UPSAMPLING_BASE_NUM_PTS))
        else:
            self.last_lid = cfg.MODEL.BOUNDARY_HEAD.NUM_DEC_LAYER
        self.npts = cfg.MODEL.BOUNDARY_HEAD.POLY_NUM_PTS

        self.mlp = MLP(self.npts*2, 256, 1, 4)
        self.mlp_infer = MLP(self.npts*2, 256, 1, 4)
        self.mlp_infer.requires_grad_(False)

        self.name = 'polycntr'

    def forward(self, verts: torch.Tensor, targets: List[Instances], lid=0):
        """verts: [num of instances over whole batch, pts, 2]"""
        device = targets.device if isinstance(targets, torch.Tensor) else targets[0].gt_boxes.device
        if lid != self.last_lid:
            return torch.tensor(0).to(device), targets

        npts = verts.shape[1]

        gt_polys = []  # list of [x0, y0, x1, y1, ...]
        for b, target in enumerate(targets):
            masks: PolygonMasks = target.get('gt_masks')
            polys_inss = masks.polygons
            for i, polys in enumerate(polys_inss):
                poly = polys[0]  # NOTE 原則的に1個しかない（稀に2個あるが）
                # align length to npts
                if npts > len(poly)/2:
                    for _ in range(npts-len(poly)//2):
                        pos = random() * (len(poly)//2-1)
                        idx = int(pos)
                        k = pos - idx
                        xm = poly[idx*2] * (1-k) + poly[idx*2+2] * k
                        ym = poly[idx*2+1] * (1-k) + poly[idx*2+3] * k
                        poly = np.concatenate([poly[:idx*2+2], [xm], [ym], poly[idx*2+2:]])
                elif npts < len(poly)/2:
                    for _ in range(len(poly)//2-npts):
                        pos = random() * (len(poly)//2)
                        idx = int(pos)
                        poly = np.concatenate([poly[:idx*2], poly[idx*2+2:]])
                poly = torch.tensor(poly).to(torch.float).to(device)
                gt_polys.append(poly)

        assert verts.shape[0] == len(gt_polys)
        gt_polys = torch.stack(gt_polys)
        verts = torch.reshape(verts, [verts.shape[0], -1])
        # print('gt polys', gt_polys.shape, 'verts', verts.shape)

        dloss = torch.mean(1 - torch.sigmoid(self.mlp(gt_polys)))
        dloss += torch.mean(torch.sigmoid(self.mlp(verts.detach())))
        dloss /= 2

        self.mlp_infer.load_state_dict(self.mlp.state_dict())
        gloss = torch.mean(1 - torch.sigmoid(self.mlp_infer(verts)))

        loss = dloss + gloss

        return loss, targets
