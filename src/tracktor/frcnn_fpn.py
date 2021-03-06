import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from collections import OrderedDict

class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes):
        self.fpn_features = None
        self.original_image_size = None
        self.image_size = None
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def predict_boxes(self, boxes, box_head=None, box_predictor=None):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        if isinstance(self.fpn_features, torch.Tensor):
            self.fpn_features = OrderedDict([(0, self.fpn_features)])

        from torchvision.models.detection.transform import resize_boxes
        boxes = resize_boxes(
            boxes, self.original_image_size[0], self.image_size[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(
            self.fpn_features, proposals, self.image_size)
        if box_head is None:
            box_head = self.roi_heads.box_head
        box_features = box_head(box_features)
        if box_predictor is None:
            box_predictor = self.roi_heads.box_predictor
        class_logits, box_regression = box_predictor(
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, self.image_size[0], self.original_image_size[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)
        targets = None
        self.original_image_size = [img.shape[-2:]]

        img, _ = self.transform(img, targets)
        self.image_size = img.image_sizes
        self.image = img.tensors[0]
        self.fpn_features = self.backbone(img.tensors)
