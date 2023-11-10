import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Instances
from boundary_former import add_boundaryformer_config
from mpvit import add_mpvit_config
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO
from compact_json import Formatter


formatter = Formatter()
formatter.max_inline_length = 200
formatter.dont_justify_numbers = True
formatter.indent_spaces = 2


class Model:
    def __init__(self, path) -> None:
        # Setup
        cfg = get_cfg()
        add_boundaryformer_config(cfg)
        add_mpvit_config(cfg)
        cfg.merge_from_file(path + '/config.yaml')
        cfg.MODEL.WEIGHTS = path + '/model_final.pth'
        cfg.MODEL.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # thr = 0.95
        # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = thr
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.  # default 0.05
        # cfg.MODEL.RPN.NMS_THRESH = 1
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000

        self.model = DefaultPredictor(cfg)

    def predict(self, img, thr=0.35):
        result: Instances = self.model(img)['instances']
        print()
        print('Result fields:', list(result.get_fields().keys()))
        boxs_xyxy = result.get('pred_boxes').tensor.cpu().numpy()  # [n_instances, 4]
        boxs_xywh = boxs_xyxy.copy()
        boxs_xywh[..., 2] -= boxs_xywh[..., 0]
        boxs_xywh[..., 3] -= boxs_xywh[..., 1]
        scores = result.get('scores').cpu().numpy()
        labels = result.get('pred_classes').cpu().numpy()

        final_polys = None
        polys_lyrs = []
        if result.has('pred_polys'):
            final_polys = self.postproc_polys(boxs_xywh, result.get('pred_polys').cpu().numpy())  # [n_instances, n_points, 2]
            i = 0
            while result.has(f'pred_polys_{i}'):
                polys_lyrs.append(self.postproc_polys(boxs_xywh, result.get(f'pred_polys_{i}').cpu().numpy()))
                i += 1

        # masks = result.get('pred_masks').cpu().numpy()
        # raw_masks = result.get('raw_masks').cpu().numpy() if result.has('raw_masks') else None

        # proposal = result._proposal
        # proposal_boxes: torch.Tensor = proposal.get('proposal_boxes').tensor
        # proposal_objectness: torch.Tensor = proposal.get('objectness_logits')
        # proposal_probs = torch.sigmoid(proposal_objectness)
        # proposal_boxes = proposal_boxes.cpu().numpy()
        # proposal_probs = proposal_probs.cpu().numpy()

        final_polys = final_polys[scores > thr]
        labels = labels[scores > thr]
        boxs_xywh = boxs_xywh[scores > thr]
        polys_lyrs = [lyr[scores > thr] for lyr in polys_lyrs]
        scores = scores[scores > thr]

        return final_polys, scores, labels, boxs_xywh, polys_lyrs

    @staticmethod
    def postproc_polys(boxs, polys):
        """Convert relative coordination polys to absolute"""
        polys = boxs[:, None, 2:] * polys + boxs[:, None, :2]
        return polys.astype(np.int)


def box_iou(box0, box1):
    ix = max(box0[0], box1[0])
    ir = min(box0[0]+box0[2], box1[0]+box1[2])
    iy = max(box0[1], box1[1])
    ib = min(box0[1]+box0[3], box1[1]+box1[3])
    i = max(0, ir-ix) * max(0, ib-iy)
    u = box0[2] * box0[3] + box1[2] * box1[3] - i
    return i/u


def match_pred_for_ann(anns, boxes):
    idxs = [i for i in range(len(boxes))]
    res = []
    for ann in anns:
        gt = ann['bbox']

        ious = [box_iou(gt, boxes[i]) for i in idxs]
        idx = idxs[np.argmax(ious)]
        if max(ious) < 0.6:
            res.append(None)
        else:
            res.append(idx)
            idxs.remove(idx)
    return res


if __name__ == '__main__':
    # data_folder = '/mnt/hd1/yoshida/mmdetection/dataset/af_data5/train/'
    data_folder = 'D:/Projects/Manga/data/Andfactory_Amtas/af_data_bubble/train'
    ann_path = data_folder + '/coco.json'
    model_folder = '../../outputs/train/af_data_bubble_train/COCO-InstanceSegmentation/sen-20.yaml'

    coco = COCO(ann_path)
    img_ids = coco.getImgIds()

    model = Model(model_folder)

    res = []
    for img_id in img_ids:
        img_path = os.path.join(data_folder, coco.loadImgs(img_id)[0]['file_name'])
        img = cv2.imread(img_path)
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)   # keys: ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'corners', 'iscrowd']

        # Predict
        final_polys, scores, labels, boxs_xywh, polys_lyrs = model.predict(img)
        pred_idxs = match_pred_for_ann(anns, boxs_xywh)

        for i, ann in enumerate(anns):
            data = {}
            data['gt'] = ann
            pred_idx = pred_idxs[i]
            if pred_idx is None:
                continue
            data['pred_box'] = list(boxs_xywh[pred_idx].astype(np.float64))
            data['pred_label'] = int(labels[pred_idx])
            data['pred_poly'] = list(np.reshape(final_polys[pred_idx], [-1]).astype(np.float64))
            data['pred_score'] = float(scores[pred_idx])
            res.append(data)
        break

    json_str = formatter.serialize(res)
    with open(data_folder+'/polys.json', 'w') as f:
        f.write(json_str)
