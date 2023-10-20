from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Instances
from boundary_former import add_boundaryformer_config
import numpy as np
import cv2
import torch


def postproc_polys(boxs, polys):
    """Convert relative coordination polys to absolute"""
    polys = boxs[:, None, 2:] * polys + boxs[:, None, :2]
    return polys.astype(np.int)


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
][7]

# Setup
cfg = get_cfg()
add_boundaryformer_config(cfg)
cfg.merge_from_file(output_folder + name + '/config.yaml')
cfg.MODEL.WEIGHTS = output_folder + name + '/model_final.pth'
cfg.MODEL.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

thr = 0.9
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = thr
cfg.MODEL.RETINANET.NMS_THRESH_TEST = thr
cfg.MODEL.RPN.NMS_THRESH = 1


model = DefaultPredictor(cfg)

img = cv2.imread('D:/Projects/Manga/data/testdata/00059.png')
# img = cv2.imread('D:/Projects/Manga/data/testdata/0003.jpg')
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
proposal = result._proposal

final_polys = postproc_polys(boxs_xywh, result.get('pred_polys').cpu().numpy())  # [n_instances, n_points, 2]
polys_lyrs = []
i = 0
while result.has(f'pred_polys_{i}'):
    polys_lyrs.append(postproc_polys(boxs_xywh, result.get(f'pred_polys_{i}').cpu().numpy()))
    i += 1
# masks = result.get('pred_masks').cpu().numpy()

print(f'Detected {polys_lyrs[0].shape[0]} instances.')


for i in range(len(boxs_xyxy)):
    box = boxs_xyxy[i].astype(int)
    score = scores[i]
    label = labels[i]
    print(f'#{i} score={score}')

    color = [0, 0, int(255 * score)]

    # Show polygon output for each level
    for lvl, polys_lyr in enumerate(polys_lyrs):
        res = img.copy() // 3
        poly = polys_lyr[i]

        cv2.rectangle(res, box[:2], box[2:], color, 1)
        cv2.drawContours(res, [poly[:, None, :]], 0, color, int(1+0*score))
        for pt in poly:
            cv2.circle(res, pt, 4, color, 1)

        cv2.imshow(f'lv.{lvl}', res)
    cv2.waitKey()
