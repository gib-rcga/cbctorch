"""
Utility functions.
"""

import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter


def fill_mask(segm, shape):
    """
    Fill a binary mask of data type np.uint8 from
    COCO style segmentations.
    """
    # Convert segmentation to list of (x, y) points
    contours = [list(zip(cont[::2], cont[1::2])) for cont in segm]

    # Round and cast contours to integer
    contours = [np.array(cont).round().astype(int) for cont in contours]

    # Create the mask and fill it
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, contours, color=1)

    return mask


def collate_fn(batch):
    return list(zip(*batch))


def encode_mask(mask):
    """
    Run-length encoding of binary mask.
    """
    # Threshold output mask from the network
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask = (mask >= 0.5).astype(np.uint8)

    # Flatten the mask, pad and encode
    pixels = np.concatenate([[0], mask.flatten(), [0]])
    runs = np.argwhere(pixels[1:] != pixels[:-1])[:, 0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def decode_mask(rle, shape=[512, 512]):
    """
    Decodes run-length encoding of binary mask.
    """
    # Get start and end points of each encoding
    assert isinstance(rle, str)
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Construct the flat mask and then reshape
    pixels = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        pixels[start:end] = 1
    mask = np.reshape(pixels, shape)

    return mask


def box_from_mask(mask):
    """
    Obtains the bounding box coordinates from a binary mask.
    """
    assert np.ndim(mask) == 2

    # Get the coordinates of foreground pixels
    coords = np.argwhere(mask)

    # Get lower and upper corner
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0) + 1
    y1, x1 = min_coords
    y2, x2 = max_coords

    # Concatenate
    box = np.array([x1, y1, x2, y2])

    # Return as float32
    return box.astype(np.float32)


def cosine_decay(epoch, init_lr, base_lr, max_epoch):
    """
    Cosine decay scheduler. Computes the multiplicative factor
    for the current learning rate.
    """
    # Fraction of lr
    alpha = base_lr / init_lr

    # Surpassed max epoch
    if epoch >= max_epoch:
        return alpha
    return alpha + 0.5 * (1 - alpha) * (1 + np.cos(np.pi * epoch / max_epoch))


def resize(image, fx, fy):
    """
    Resizes an image and crops or pads it.
    """
    # Get original dimensions
    h, w = image.shape[:2]

    # Resize the image
    image = cv2.resize(image.astype(np.float32), (0, 0), fx=fx, fy=fy)
    h_new, w_new = image.shape[:2]

    # Handle padding or cropping
    if h_new > h:
        h_0 = h_new // 2 - h // 2
        image = image[h_0 : h_0 + h, :]
    else:
        pad_0 = (h - h_new) // 2
        pad_1 = h - h_new - pad_0
        if image.ndim == 3:
            image = np.pad(image, ((pad_0, pad_1), (0, 0), (0, 0)))
        else:
            image = np.pad(image, ((pad_0, pad_1), (0, 0)))
    if w_new > w:
        w_0 = w_new // 2 - w // 2
        image = image[:, w_0 : w_0 + w]
    else:
        pad_0 = (w - w_new) // 2
        pad_1 = w - w_new - pad_0
        if image.ndim == 3:
            image = np.pad(image, ((0, 0), (pad_0, pad_1), (0, 0)))
        else:
            image = np.pad(image, ((0, 0), (pad_0, pad_1)))

    return image


def compute_iou(boxes_1, boxes_2):
    """
    Pairwise IoU between two arrays of boxes formatted as [x1, y1, x2, y2].
    """
    # Ensure np.arrays
    boxes_1 = np.asarray(boxes_1)
    boxes_2 = np.asarray(boxes_2)

    # Split the boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = np.split(boxes_1, 4, axis=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = np.split(boxes_2, 4, axis=-1)

    # Get intersection coordinates of each pair and the area
    x1 = np.maximum(b1_x1, np.transpose(b2_x1))
    y1 = np.maximum(b1_y1, np.transpose(b2_y1))
    x2 = np.minimum(b1_x2, np.transpose(b2_x2))
    y2 = np.minimum(b1_y2, np.transpose(b2_y2))
    inter_w = np.maximum(x2 - x1, 0)
    inter_h = np.maximum(y2 - y1, 0)
    inter_area = inter_h * inter_w

    # Get the area of all boxes
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)

    # Union area is the sum of both areas minus the intersection
    union_area = b1_area + np.transpose(b2_area) - inter_area

    # Intersection over union
    return inter_area / union_area


def label_nms(boxes, scores, labels, iou_thresh=0.5, joint_labels=[1, 2]):
    """
    Per label non-maximum suppression (NMS). Labels in the joint labels
    list are considered a single label in NMS.

    Returns the indices of the kept boxes.
    """
    # Return empty list if no boxes are given
    if len(boxes) == 0:
        return []

    # Create index list for the labels
    idxs = [joint_labels]
    idxs = idxs + [i for i in range(1, np.max(labels) + 1) if i not in joint_labels]

    # Iterate over labels and perform NMS
    keep = []
    for idx in idxs:
        label_idxs = np.argwhere(np.isin(labels, idx))[:, 0]
        keep_split = nms(boxes[label_idxs], scores[label_idxs], iou_thresh=iou_thresh)
        keep += list(label_idxs[keep_split])

    return keep


def nms(boxes, scores, iou_thresh=0.5):
    """
    Non-maximum supression.
    """
    # Return empty list if no boxes are given
    if len(boxes) == 0:
        return []

    # Sort boxes in descending scores
    idxs = np.argsort(scores)[::-1]

    # Iterate and keep indices
    keep = []
    while idxs.size > 0:
        # Keep box with highest score
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break

        # Keep only those that do not overlap with the box kept
        iou = np.squeeze(compute_iou(boxes[i], boxes[idxs[1:]]), axis=0)
        inds = np.argwhere(iou <= iou_thresh)[:, 0]
        idxs = idxs[inds + 1]

    return keep


def evaluate_model(model, data_loader, device, log_dir):
    """
    Function to evaluate a trained model on a validation dataset.
    """
    # If log dir does not exist, create
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set model in validation mode
    model.eval()

    # Send model to device
    model.to(device)

    # Construct target and detection datasets for json
    targets_list = []
    detections_list = []

    with torch.no_grad():
        # Iterate over the data loader
        pbar = tqdm(data_loader, desc="Evaluation")
        for batch in pbar:
            # Unpack batch
            images, targets = batch

            # Process targets before sending to GPU
            targets_proc = [{k: v.tolist() for k, v in t.items()} for t in targets]
            # Encode the mask with RLE as a string
            for target in targets_proc:
                target["masks"] = [encode_mask(m) for m in target["masks"]]

            # Append to targets list
            targets_list += targets_proc

            # Send the elements in the batch to the device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get targets List[Dict] and format them
            detections = model(images, targets)

            # Send detections to the CPU and express as lists
            detections = [
                {k: v.cpu().tolist() for k, v in d.items()} for d in detections
            ]
            # Encode the mask with RLE as a string
            for i, detection in enumerate(detections):
                detection["masks"] = [encode_mask(m) for m in detection["masks"]]
                # Add the image id to the detections
                detection["image_id"] = targets_proc[i]["image_id"]

            # Append to detections list
            detections_list += detections

    # Close progress bar
    pbar.close()

    # Create the json dictionary
    json_dict = {
        "targets": targets_list,
        "detections": detections_list,
    }

    # Dump detections into json
    json_path = os.path.join(log_dir, "evaluation.json")
    with open(json_path, "w") as fp:
        json.dump(json_dict, fp)

    print("Evaluation done. Targets and detections saved at {}.".format(json_path))


class EarlyStopping:
    """
    Early stopping for training.
    """

    def __init__(self, mode="min", delta=1e-3, patience=10):
        # Declare arguments
        assert mode in ("min", "max")
        self.mode = mode

        # Best metric
        self.best = np.inf
        if self.mode == "max":
            self.best *= -1

        # Smallest change to consider valid update
        self.delta = delta

        # Patience until stop
        self.patience = patience
        self.wait_count = 0

    def step(self, metric):
        # Check metric and if its new best, save weights
        if self.has_improved(metric):
            self.best = metric
            self.wait_count = 0
        else:
            self.wait_count += 1

        # Return True only when patience is reached
        return self.wait_count >= self.patience

    def has_improved(self, metric):
        # Different checks depending on mode
        if self.mode == "max":
            improved = metric > self.best - self.delta
        else:
            improved = metric < self.best + self.delta

        return improved


class Trainer:
    """
    Class to simplify training loop.
    """

    def __init__(
        self,
        model,
        optimizer,
        epochs,
        train_data_loader,
        accum_iters=1,
        val_data_loader=None,
        lr_scheduler=None,
        scheduler_type="epoch",
        log_dir="./training_logs",
        run_name=None,
        fold=None,
        save_dir="./saved_models",
        save_eval=False,
        split=None,
    ):
        # Save training variables
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_grad = 5.0  # Clip gradient norm
        self.accum_iters = accum_iters  # Gradient accumulation
        self.save_eval = save_eval

        # Early stopping handler
        self.early_stop = EarlyStopping(patience=20)

        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        assert scheduler_type in ("epoch", "iter")
        self.scheduler_type = scheduler_type

        # Save data_loaders
        self.train_dl = train_data_loader
        self.val_dl = val_data_loader

        # If not given a name, save as date_time
        if run_name is None:
            now = datetime.now()
            run_name = now.strftime("%d%m%Y_%H%M%S")

        # Directory and writer for logging training
        self.run_log_dir = os.path.join(log_dir, run_name)
        self.run_save_dir = os.path.join(save_dir, run_name)
        model_name = "mrcnn"
        # If doing k-fold cross validation, create subdirectories for folds
        if fold is not None:
            assert isinstance(fold, int)
            self.run_log_dir = os.path.join(self.run_log_dir, f"fold_{fold}")
            model_name = model_name + f"_fold_{fold}"
        if split is not None:
            self.run_log_dir = os.path.join(self.run_log_dir, f"split_{split}")
        self.split = split

        # Create directory
        Path(self.run_log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_log_dir)

        # Directory for saving the trained model weights
        Path(self.run_save_dir).mkdir(parents=True, exist_ok=True)
        self.save_path = os.path.join(self.run_save_dir, model_name + ".pth")

    def fit(self):
        """
        Training loop.
        """
        # Print split
        if self.split is not None:
            print(f"Training split {self.split}:")

        # Send model to device
        self.model.to(self.device)

        # Iterate over epochs
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            # Progress bar formatting
            pbar.set_description(f"Epoch {epoch}")

            # Train epoch
            train_loss = self.train_epoch(epoch)
            losses = {}
            losses["train_loss"] = train_loss

            # If validation data loader is given evaluate
            with torch.no_grad():
                if self.val_dl:
                    val_loss = self.val_epoch(epoch)
                    losses["val_loss"] = val_loss

                    # Check for early stopping, if returns True break
                    if self.early_stop.step(val_loss):
                        break

            # Update the progress bar with losses
            pbar.set_postfix(losses)

        # Close the TensorBoard writer and progress bar
        self.writer.close()
        pbar.close()

        # Output message
        print(f"Finished at epoch {epoch}.")

        # Evaluate the model if validation data loader is given
        with torch.no_grad():
            if self.val_dl and self.save_eval:
                evaluate_model(
                    model=self.model,
                    data_loader=self.val_dl,
                    device=self.device,
                    log_dir=self.run_log_dir,
                )

        # Save the model's weights
        torch.save(self.model.state_dict(), self.save_path)

    def train_epoch(self, epoch):
        """
        Train a single epoch.
        """
        # Set model in training mode
        self.model.train()

        # Create epoch losses accumulator
        epoch_losses = defaultdict(lambda: 0)

        # Iterate over the data loader
        pbar = tqdm(self.train_dl, desc="Training", leave=False)
        loss_display = tqdm(desc="Metrics", bar_format="{desc}{postfix}", leave=False)
        for i, batch in enumerate(pbar):
            # Send the elements in the batch to the device
            images, targets = batch
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Perform forward pass
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())  # Aggregate losses

            # Accumulate the epoch losses
            epoch_losses["loss"] += loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()

            # Divide loss by accumulations to normalize contributions
            loss = loss / self.accum_iters

            # Perform backward pass
            loss.backward()

            # Clip the gradient norm
            params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad)

            # Update weights
            if ((i + 1) % self.accum_iters == 0) or (i + 1 == len(self.train_dl)):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # LR scheduling
            if self.lr_scheduler and self.scheduler_type == "iter":
                self.lr_scheduler.step()

            # Print the current running losses in the progress bar
            loss_display.set_postfix({k: v / (i + 1) for k, v in epoch_losses.items()})

        # LR scheduling
        if self.lr_scheduler and self.scheduler_type == "epoch":
            self.lr_scheduler.step()

        # Get the mean losses
        epoch_losses = {f"train/{k}": v / (i + 1) for k, v in epoch_losses.items()}

        # Log the losses
        for k, v in epoch_losses.items():
            self.writer.add_scalar(k, v, epoch)

        return epoch_losses["train/loss"]

    def val_epoch(self, epoch):
        """
        Evaluation for Mask R-CNN torchvision models.
        """
        # Set the model as training, to output the losses
        self.model.train()

        # Create epoch losses accumulator
        epoch_losses = defaultdict(lambda: 0)

        # Iterate over the data loader
        pbar = tqdm(self.val_dl, desc="Validation", leave=False)
        loss_display = tqdm(desc="Metrics", bar_format="{desc}{postfix}", leave=False)
        for i, batch in enumerate(pbar):
            # Send the elements in the batch to the device
            images, targets = batch
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Perform forward pass
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())  # Aggregate losses

            # Accumulate the epoch losses
            epoch_losses["loss"] += loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()

            # Print the current running losses in the progress bar
            loss_display.set_postfix({k: v / (i + 1) for k, v in epoch_losses.items()})

        # Get the mean losses
        epoch_losses = {f"val/{k}": v / (i + 1) for k, v in epoch_losses.items()}

        # Log the losses
        for k, v in epoch_losses.items():
            self.writer.add_scalar(k, v, epoch)

        return epoch_losses["val/loss"]
