"""
Mask R-CNN using torchvision's implementation.
"""

import torch.nn as nn
import torch.functional as F
import torchvision.models.detection as models
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import *
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision._internally_replaced_utils import load_state_dict_from_url

from .data import CBCTDataset


def dice_loss(mask_preds, mask_targets):
    # Mask logits as probs
    mask_preds = torch.sigmoid(mask_preds)

    # Flatten all dims except first
    mask_preds = torch.flatten(mask_preds, start_dim=1)
    mask_targets = torch.flatten(mask_targets, start_dim=1)

    # Intersection
    inter = torch.sum(mask_preds * mask_targets, dim=1)
    preds = torch.sum(mask_preds, dim=1)
    targets = torch.sum(mask_targets, dim=1)

    # DSC
    dice = (2 * inter + 1) / (preds + targets + 1)

    return torch.mean(1 - dice)


def tooth_classification_loss(tooth_logits, gt_tooth_labels, matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor]) -> Tensor
    """
    Computes the loss for tooth classification.

    Args:
        tooth_logits (Tensor)
        gt_tooth_labels (List[Tensor])
        matched_idxs (Tensor)

    Returns:
        loss (Tensor)
    """

    tooth_labels = [
        gt_label[idxs] for gt_label, idxs in zip(gt_tooth_labels, matched_idxs)
    ]
    tooth_labels = torch.cat(tooth_labels, dim=0)

    return F.cross_entropy(tooth_logits, tooth_labels)


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    # Mask predictions
    mask_preds = mask_logits[
        torch.arange(labels.shape[0], device=labels.device), labels
    ]

    # Binary cross-entropy loss
    bce = F.binary_cross_entropy_with_logits(mask_preds, mask_targets)

    # Dice loss
    dl = dice_loss(mask_preds, mask_targets)

    return bce + dl


def get_optimizer_params(model):
    """
    Returns the base (COCO pretrained) and randomly initialized parameters,
    separately.
    """
    # New layers
    new_layers = ("tooth_head", "tooth_predictor", "mask_predictor", "box_predictor")

    # Get params
    assert isinstance(model, nn.Module)
    base_params = [
        param
        for name, param in model.named_parameters()
        if not any(layer in name for layer in new_layers) and param.requires_grad
    ]
    rand_params = [
        param
        for name, param in model.named_parameters()
        if any(layer in name for layer in new_layers) and param.requires_grad
    ]

    return base_params, rand_params


def get_model(dataset, model_type="full", trainable_layers=0):
    """
    Function to construct the Mask R-CNN model.
    """
    assert isinstance(dataset, CBCTDataset)
    # Anchor sizes adjusted to the percentiles of our dataset
    num_teeth = len(dataset.tooth_ids) + 1  # Num of tooth IDs and background
    representation_size = 1024
    anchor_sizes = ((16,), (32,), (64,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = models.faster_rcnn.AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    assert model_type in ("full", "segm")
    if model_type == "full":
        has_teeth = True
    else:
        has_teeth = False

    # Construct the model
    backbone = resnet_fpn_backbone(
        "resnet50", pretrained=False, trainable_layers=trainable_layers
    )
    model = models.MaskRCNN(
        backbone=backbone,
        num_classes=91,
        min_size=dataset.min_size,
        max_size=dataset.max_size,
        rpn_anchor_generator=rpn_anchor_generator,
    )

    # Get feature map parameters
    out_channels = model.backbone.out_channels

    # Greater resolution in Mask RoIAlign
    resolution = 28
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=resolution, sampling_ratio=2
    )

    # Define new teeth layers
    fc_neurons = 512
    tooth_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((resolution // 4)),
        models.faster_rcnn.TwoMLPHead(
            (resolution // 4) ** 2 * out_channels, fc_neurons
        ),
    )
    tooth_predictor = ToothPredictor(fc_neurons, num_teeth)

    # Create new roi_heads
    roi_heads = CBCTRoIHeads(
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=model.roi_heads.box_predictor,
        # Faster R-CNN training
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        # Faster R-CNN inference
        score_thresh=0.5,
        nms_thresh=0.5,
        detections_per_img=100,
        # Mask
        mask_roi_pool=mask_roi_pool,
        mask_head=model.roi_heads.mask_head,
        mask_predictor=model.roi_heads.mask_predictor,
        # Tooth
        tooth_head=tooth_head,
        tooth_predictor=tooth_predictor,
        has_teeth=has_teeth,
    )

    # Replace roi_heads
    model.roi_heads = roi_heads

    # Load COCO weights with strict=False
    model_urls = {
        "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
    }
    state_dict = load_state_dict_from_url(
        model_urls["maskrcnn_resnet50_fpn_coco"], progress=True
    )
    model.load_state_dict(state_dict, strict=False)

    # Re-initialize the box (and label) and mask predictors
    box_predictor = models.faster_rcnn.FastRCNNPredictor(
        representation_size, dataset.num_classes
    )
    mask_predictor = models.mask_rcnn.MaskRCNNPredictor(
        out_channels, out_channels // 2, dataset.num_classes
    )

    # Replace the layers in the model
    model.roi_heads.box_predictor = box_predictor
    model.roi_heads.mask_predictor = mask_predictor

    return model


class ToothPredictor(nn.Module):
    """
    Predictor for tooth label.

    Args:
        in_channels (int): number of input channels
        num_teeth (int): number of output tooth labels (including background),
        default is FDI number of adult teeth + background
    """

    def __init__(self, in_channels, num_teeth):
        super().__init__()
        self.tooth_fc = nn.Linear(in_channels, num_teeth)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        tooth_logits = self.tooth_fc(x)

        return tooth_logits


class CBCTRoIHeads(RoIHeads):
    """
    Adaptation of the Mask R-CNN RoIHeads to include a tooth classifier.
    """

    def __init__(self, tooth_head, tooth_predictor, has_teeth=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_teeth = has_teeth
        if self.has_teeth:
            self.tooth_head = tooth_head
            self.tooth_predictor = tooth_predictor
        else:
            self.tooth_head = None
            self.tooth_predictor = None

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(
                        f"target boxes must of float type, instead got {t['boxes'].dtype}"
                    )
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(
                        f"target labels must of int64 type, instead got {t['labels'].dtype}"
                    )
                if not t["tooth_labels"].dtype == torch.int64:
                    raise TypeError(
                        f"target tooth labels must of int64 type, instead got {t['labels'].dtype}"
                    )
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(
                            f"target keypoints must of float type, instead got {t['keypoints'].dtype}"
                        )

        if self.training:
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
            ) = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        # Incorporate tooth branch alongside the mask
        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                # RoIAlign
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes
                )

                if self.has_teeth:
                    # Pass the mask features also through the tooth head
                    tooth_features = self.tooth_head(mask_features)
                    tooth_logits = self.tooth_predictor(tooth_features)

                # Mask branch
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError(
                        "targets, pos_matched_idxs, mask_logits cannot be None when training"
                    )

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = {"loss_mask": rcnn_loss_mask}

                if self.has_teeth:
                    gt_tooth_labels = [t["tooth_labels"] for t in targets]
                    loss_tooth = tooth_classification_loss(
                        tooth_logits, gt_tooth_labels, pos_matched_idxs
                    )
                    loss_mask["loss_tooth"] = loss_tooth

            else:
                # Masks
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

                # Teeth
                if self.has_teeth:
                    labels_per_image = [label.shape[0] for label in labels]
                    tooth_preds = torch.argmax(tooth_logits, dim=-1)
                    tooth_preds = tooth_preds.split(labels_per_image, dim=0)
                    for tooth_pred, r in zip(tooth_preds, result):
                        r["tooth_labels"] = tooth_pred

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes
            )
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError(
                        "both targets and pos_matched_idxs should not be None when in training mode"
                    )

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals
                )
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
