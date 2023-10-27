from random import random
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Instances
from boundary_former import add_boundaryformer_config
from mpvit import add_mpvit_config
import numpy as np
import cv2
import torch


def postproc_polys(boxs, polys):
    """Convert relative coordination polys to absolute"""
    polys = boxs[:, None, 2:] * polys + boxs[:, None, :2]
    return polys.astype(np.int)


def contain_xy(box_xyxy, x, y):
    return box_xyxy[0] < x < box_xyxy[2] and box_xyxy[1] < y < box_xyxy[3]


def in_xyxy(box_xyxy, x, y, X, Y):
    box = box_xyxy
    return x < box[0] and y < box[1] and X > box[2] and Y > box[3]


def hsv2rgb(h, s, v):
    c = v * s
    x = c * (1-abs((h // 60) % 2 - 1))
    m = v - c
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    elif 300 <= h < 360:
        r, g, b = c, 0, x
    return int((r+m)*255), int((g+m)*255), int((b+m)*255)


def rand_rgb(v):
    return hsv2rgb(random()*360, random()*0.4+0.6, v)


output_folder = '../../outputs/train/af_data_bubble_train/COCO-InstanceSegmentation/'
name = [
    'sen-1.yaml',  # (0) 64pts Vanilla
    'sen-2.yaml',  # (1) 64pts Residual
    'sen-3.yaml',  # (2) 8pts Vanilla: Better on 8pts level
    'sen-4.yaml',  # (3) 64pts DICE_LOSS_WEIGHT_FP=10
    'sen-5.yaml',  # (4) 64pts DETACH_EACH_LEVEL
    'sen-6.yaml',  # (5) 64pts DICE_LOSS_WEIGHT_FN=10
    'sen-7.yaml',  # (6) 8pts 4levels UPSAMPLING=False DEEP_SUPERVISION=False (i.e. only supervise the last level)
    'sen-8.yaml',  # (7) 8pts 1levels 4layers_per_level UPSAMPLING=False DEEP_SUPERVISION=False
    'sen-9.yaml',  # (8) 8pts 4levels UPSAMPLING=False
    'sen-10.yaml',  # (9) 32pts 3levels 2layers_per_level DEEP_SUPERVISION=False
    'sen-11.yaml',  # (10) 32pts 3levels 2layers_per_level
    'sen-12.yaml',  # (11) 32pts
    'sen-13.yaml',  # (12) 64pts box_reg 10 beta 0.5
    'sen-14.yaml',  # (13) 64pts box_reg 100 beta 0.5
    'sen-15.yaml',  # (14) 64pts MPViT
][-1]

# Setup
cfg = get_cfg()
add_boundaryformer_config(cfg)
add_mpvit_config(cfg)
cfg.merge_from_file(output_folder + name + '/config.yaml')
cfg.MODEL.WEIGHTS = output_folder + name + '/model_final.pth'
cfg.MODEL.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# thr = 0.99
# cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = thr
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.  # default 0.05
# cfg.MODEL.RPN.NMS_THRESH = 1
cfg.TEST.DETECTIONS_PER_IMAGE = 1000


model = DefaultPredictor(cfg)

img = cv2.imread('D:/Projects/Manga/data/testdata/0003.jpg')
# img = cv2.imread('D:/Projects/Manga/data/testdata/00059.png')
# img = cv2.imread('D:/Projects/Manga/data/testdata/00016.png')
# img = cv2.imread('D:/Projects/Manga/data/testdata/00053.png')


# Inference
result: Instances = model(img)['instances']
print()
print('Result fields:', list(result.get_fields().keys()))
boxs_xyxy = result.get('pred_boxes').tensor.cpu().numpy()  # [n_instances, 4]
boxs_xywh = boxs_xyxy.copy()
boxs_xywh[..., 2] -= boxs_xywh[..., 0]
boxs_xywh[..., 3] -= boxs_xywh[..., 1]
scores = result.get('scores').cpu().numpy()
labels = result.get('pred_classes').cpu().numpy()

final_polys = postproc_polys(boxs_xywh, result.get('pred_polys').cpu().numpy())  # [n_instances, n_points, 2]
polys_lyrs = []
i = 0
while result.has(f'pred_polys_{i}'):
    polys_lyrs.append(postproc_polys(boxs_xywh, result.get(f'pred_polys_{i}').cpu().numpy()))
    i += 1
# masks = result.get('pred_masks').cpu().numpy()

proposal = result._proposal
proposal_boxes: torch.Tensor = proposal.get('proposal_boxes').tensor
proposal_objectness: torch.Tensor = proposal.get('objectness_logits')
proposal_probs = torch.sigmoid(proposal_objectness)
proposal_boxes = proposal_boxes.cpu().numpy()
proposal_probs = proposal_probs.cpu().numpy()


print(f'Detected {polys_lyrs[0].shape[0]} instances, from {len(proposal_probs)} proposals.')


# ############ Proposals

res = img.copy() // 3
for box, score in zip(proposal_boxes, proposal_probs):
    # if not (0.9 < score < 1):
    #     continue
    if not contain_xy(box, 618, 165):
        continue
    if not in_xyxy(box, 519, 66, 745, 263):
        continue

    box = box.astype(int)
    print(score, box)
    color = rand_rgb(score)

    cv2.rectangle(res, box[:2], box[2:], color, 1)
cv2.imshow('proposal', res)
cv2.waitKey()


# ############ For each instance

# res = img.copy() // 3
# for i in range(len(boxs_xyxy)):
#     box = boxs_xyxy[i].astype(int)
#     score = scores[i]
#     label = labels[i]
#     # if not contain_xy(box, 618, 165):
#     #     continue
#     print(f'#{i} score={score}')

#     color = rand_rgb(score)

#     # Show last output
#     polys_lyr = polys_lyrs[-1]
#     poly = polys_lyr[i]

#     if SHOWBOX:
#         cv2.rectangle(res, box[:2], box[2:], color, 1)
#     cv2.drawContours(res, [poly[:, None, :]], 0, color, int(1+0*score))
#     for pt in poly:
#         cv2.circle(res, pt, 4, color, 1)

#     cv2.imshow(f'Ins {i}', res)
#     # cv2.waitKey()
# cv2.waitKey()


# ############ For each level

SHOWBOX = False
for lvl, polys_lyr in enumerate(polys_lyrs):
    res = img.copy() // 3
    for i in range(len(boxs_xyxy)):
        box = boxs_xyxy[i].astype(int)
        score = scores[i]
        label = labels[i]
        poly = polys_lyr[i]

        # if not contain_xy(box, 618, 165):
        #     continue
        print(f'#{i} score={score}')

        color = rand_rgb(score)

        if SHOWBOX:
            cv2.rectangle(res, box[:2], box[2:], color, 1)
        cv2.drawContours(res, [poly[:, None, :]], 0, color, int(1+0*score))
        for pt in poly:
            cv2.circle(res, pt, 4, color, 1)

    cv2.imshow(f'lv.{lvl}', res)
cv2.waitKey()
