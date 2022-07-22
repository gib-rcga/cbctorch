"""
Transformations and augmentations for data.
"""

import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from skimage.restoration import inpaint_biharmonic

from .utils import box_from_mask, resize


def switch_tooth(tooth_id):
    """
    When performing horizontal flip augmentation, change
    the tooth ID to its sagittal opposite.
    """
    # Find the value of the 2 integer place
    quadrant = tooth_id // 10

    # If its either 2 or 4 substract 10 else add 10
    if quadrant % 2 == 0:
        tooth_id -= 10
    else:
        tooth_id += 10

    return tooth_id


class Compose:
    """
    Concatenate transforms and apply one after the other.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        # Apply all transforms
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor:
    """
    Transform to convert the image and targets into:
    - image (FloatTensor[C, H, W]) in range [0, 1]
    - target
        - boxes (FloatTensor[N, 4]) with 0 <= x1 < x2 <= W
        and 0 <= y1 < y2 <= H
        - labels (Int64Tensor[N])
        - tooth_labels (Int64Tensor[N])
        - masks (UInt8Tensor[N, H, W]) either 0 or 1
        - image_id (Int64Tensor)
    """

    def __call__(self, image, target):
        # Use torchvision transform to scale and transform image
        image = to_tensor(image)

        # Transform the targets
        assert isinstance(target, dict)
        target = {k: torch.as_tensor(v) for k, v in target.items()}

        return image, target


class Reshape:
    """
    Reshapes the image to be (H, W) = (512, 512), also adapts
    mask and bounding boxes.
    """

    def __call__(self, image, target):
        # Image shape and scale for boxes
        img_shape = image.shape[:2]
        img_scale = np.concatenate([img_shape, img_shape], axis=0)
        out_scale = np.array([512, 512, 512, 512])

        # Get resize factor so that max dimension is 512
        fs = 512 / np.max(img_shape)
        out_w = np.minimum(512, round(img_shape[1] * fs))
        out_h = np.minimum(512, round(img_shape[0] * fs))

        # Resize the image and masks
        image = cv2.resize(image, dsize=(out_w, out_h))
        res_masks = []
        for m in target["masks"]:
            res_masks.append(cv2.resize(m.astype(np.float32), (out_w, out_h)))
        res_masks = np.asarray(res_masks)

        # Pad to be (512, 512)
        h_pad_0 = (512 - image.shape[0]) // 2
        h_pad_1 = 512 - image.shape[0] - h_pad_0
        w_pad_0 = (512 - image.shape[1]) // 2
        w_pad_1 = 512 - image.shape[1] - w_pad_0
        image = np.pad(image, ((h_pad_0, h_pad_1), (w_pad_0, w_pad_1), (0, 0)))
        res_masks = np.pad(res_masks, ((0, 0), (h_pad_0, h_pad_1), (w_pad_0, w_pad_1)))
        # Express masks as binary
        res_masks = (res_masks > 0.5).astype(np.uint8)

        # Adapt the bounding boxes
        res_boxes = np.array([b * out_scale / img_scale for b in target["boxes"]])

        # Add targets
        target["boxes"] = res_boxes
        target["masks"] = res_masks

        return image, target


class RandomScale:
    """
    Scale or downscale the image and targets a random percent in both axis.
    Args:
    - max_pct: Maximum percentage to scale the image.
    """

    def __init__(self, max_pct=20):
        self.max_pct = max_pct

    def __call__(self, image, target):
        # Obtain the random pct
        pct = np.random.uniform(-self.max_pct, self.max_pct)
        fs = 1 + pct / 100

        # Scale all the masks
        scl_masks = [resize(m, fx=fs, fy=fs) for m in target["masks"]]
        # Convert masks back to binary after the interpolation
        scl_masks = np.array([(m > 0.5).astype(np.uint8) for m in scl_masks])
        # Handle cases where mask was scaled out of the image
        non_zero = np.argwhere([np.any(m) for m in scl_masks])[:, 0]
        scl_masks = scl_masks[non_zero, ...]

        # It this transform removes all labels return original image and target
        if len(non_zero) == 0:
            return image, target

        # Scale the image
        image = resize(image, fx=fs, fy=fs)

        # Get the new bounding boxes
        scl_boxes = np.array([box_from_mask(m) for m in scl_masks])

        # Build targets
        target["labels"] = target["labels"][non_zero]
        target["tooth_labels"] = target["tooth_labels"][non_zero]
        target["boxes"] = scl_boxes
        target["masks"] = scl_masks

        return image, target


class RandomShift:
    """
    Translate an image randomly in the X and Y axis an amount of pixels
    determined by a random percentage of that axis.
    Args:
    - max_pct: Maximum translation percentage.
    """

    def __init__(self, max_pct=5):
        self.max_pct = max_pct

    def __call__(self, image, target):
        # Obtain the random pct for x and y
        pct_x = np.random.uniform(-self.max_pct, self.max_pct)
        pct_y = np.random.uniform(-self.max_pct, self.max_pct)
        fx = pct_x / 100
        fy = pct_y / 100

        # Get the pixel translations
        h, w = image.shape[:2]
        tx = round(fx * w)
        ty = round(fy * h)
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

        # Shift all the masks
        shf_masks = [cv2.warpAffine(m, M, (w, h)) for m in target["masks"]]
        # Convert masks back to binary after the interpolation
        shf_masks = np.array([(m > 0.5).astype(np.uint8) for m in shf_masks])
        # Handle cases where mask was shifted out of the image
        non_zero = np.argwhere([np.any(m) for m in shf_masks])[:, 0]
        shf_masks = shf_masks[non_zero, ...]

        # It this transform removes all labels return original image and target
        if len(non_zero) == 0:
            return image, target

        # Scale the image
        image = cv2.warpAffine(image, M, (w, h))

        # Get the new bounding boxes
        shf_boxes = np.array([box_from_mask(m) for m in shf_masks])

        # Build targets
        target["labels"] = target["labels"][non_zero]
        target["tooth_labels"] = target["tooth_labels"][non_zero]
        target["boxes"] = shf_boxes
        target["masks"] = shf_masks

        return image, target


class RandomRot:
    """
    Rotate image and get rotated targets.
    Args:
    - max_theta: Maximum rotation angle in degrees.
    """

    def __init__(self, max_theta=10):
        self.max_theta = max_theta

    def __call__(self, image, target):
        # Obtain the random rotation angle
        theta = np.random.uniform(-self.max_theta, self.max_theta)

        # Get the original image dimensions
        h, w = np.shape(image)[:2]
        center = (w // 2, h // 2)

        # Rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, theta, 1)

        # Rotate masks
        rot_masks = [cv2.warpAffine(m, rot_mat, (w, h)) for m in target["masks"]]
        # Convert masks back to binary after the interpolation
        rot_masks = np.array([(m > 0.5).astype(np.uint8) for m in rot_masks])
        # Handle cases where mask was rotated out of the image
        non_zero = np.argwhere([np.any(m) for m in rot_masks])[:, 0]
        rot_masks = rot_masks[non_zero, ...]

        # It this transform removes all labels return original image and target
        if len(non_zero) == 0:
            return image, target

        # Rotate image
        image = cv2.warpAffine(image, rot_mat, (w, h))

        # Obtain the new bounding boxes from the rotated masks
        rot_boxes = np.array([box_from_mask(m) for m in rot_masks])

        # Build rotated targets
        target["labels"] = target["labels"][non_zero]
        target["tooth_labels"] = target["tooth_labels"][non_zero]
        target["boxes"] = rot_boxes
        target["masks"] = rot_masks

        return image, target


class RandomHFlip:
    """
    Random horizontal flip of the image.
    Args:
    - tooth_ids: Dictionary that maps tooth IDs to
    labels in the dataset.
    - prob: Probability of the flip.
    """

    def __init__(self, tooth_ids, prob=0.5):
        assert isinstance(tooth_ids, dict)
        self.id_to_label = tooth_ids
        self.label_to_id = {v: k for k, v in tooth_ids.items()}
        self.prob = prob

    def __call__(self, image, target):
        # Generate random state
        if np.random.uniform() > self.prob:
            return image, target

        # Horizontal flip
        assert np.ndim(image) == 3
        # Copy inverted views to avoid negative indexing error
        image = np.copy(image[:, ::-1, :])
        masks = [np.copy(m[:, ::-1]) for m in target["masks"]]

        # Get new boxes
        boxes = [box_from_mask(m) for m in masks]

        # If flipped, change the tooth labels accordingly
        tooth_ids = [
            switch_tooth(self.label_to_id[l]) if l != 0 else 0
            for l in target["tooth_labels"]
        ]
        tooth_labels = [self.id_to_label[i] if i != 0 else 0 for i in tooth_ids]

        # Build flipped targets
        target["boxes"] = np.array(boxes)
        target["masks"] = np.array(masks)
        target["tooth_labels"] = np.array(tooth_labels)

        return image, target


class RandomRotDenture:
    """
    Rotates each tooth in an image (teeth and metal) a random amount.
    Args:
    - max_theta: Maximum rotation angle in degrees.
    """

    def __init__(self, max_theta=30):
        self.max_theta = max_theta

    def __call__(self, image, target):
        # Get the teeth indices
        tooth_idxs = np.argwhere(target["tooth_labels"] > 0)[:, 0]

        # Convert image to normalized float32
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Generate random states for each tooth
        augment = np.random.uniform(size=len(tooth_idxs)) <= 0.5

        for i, idx in enumerate(tooth_idxs):
            if augment[i]:
                # Perform augmentation
                box = target["boxes"][idx]
                mask = target["masks"][idx]
                image, rot_box, rot_mask = self.rotate_tooth(image, box, mask)

                # Update targets
                target["boxes"][idx] = rot_box
                target["masks"][idx] = rot_mask

        return image, target

    def rotate_tooth(self, image, box, mask):
        # Get random rotation angle
        theta = np.random.uniform(-self.max_theta, self.max_theta)

        # Operate only on the first channel (grayscale 3-channel)
        h, w = np.shape(image)[:2]
        image = image[..., 0]

        # Dilate the mask
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Get the tooth and erase it from the image
        tooth = np.zeros_like(image)
        tooth[mask > 0] = np.copy(image[mask > 0])
        image[mask > 0] = 0

        # Get the mask center from its bounding box
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        c1 = round(x1 + 0.5 * box_w)
        c2 = round(y1 + 0.5 * box_h)
        center = (c1, c2)

        # Rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, theta, 1)

        # Rotate the tooth and mask
        rot_tooth = cv2.warpAffine(tooth, rot_mat, (w, h))
        rot_mask = cv2.warpAffine(mask, rot_mat, (w, h))

        # Erode the mask back and put the tooth into the image
        rot_mask = (rot_mask > 0.5).astype(np.uint8)
        rot_mask = cv2.erode(rot_mask, kernel, iterations=1)
        image[rot_mask > 0] = rot_tooth[rot_mask > 0]

        # Inpaint the gap between the dilated and eroded rotated masks
        gap = np.logical_xor(rot_mask > 0, mask > 0).astype(np.uint8)
        image = inpaint_biharmonic(image, gap)
        # Transform to float32 (inpaint returns float64)
        image = image.astype(np.float32)
        image = np.repeat(image[..., None], repeats=3, axis=-1)

        # Get new box from rotated mask
        rot_box = box_from_mask(rot_mask)

        return image, rot_box, rot_mask


class RandomScaleDenture:
    """
    Randomly scales each tooth up or down based on a random percentage.
    Args:
    - max_pct: maximum percentage to scale up or down.
    """

    def __init__(self, max_pct=20):
        self.max_pct = max_pct

    def __call__(self, image, target):
        # Get the teeth indices
        tooth_idxs = np.argwhere(target["tooth_labels"] > 0)[:, 0]

        # Convert image to normalized float32
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Generate random states for each tooth
        augment = np.random.uniform(size=len(tooth_idxs)) <= 0.5

        for i, idx in enumerate(tooth_idxs):
            if augment[i]:
                # Perform augmentation
                box = target["boxes"][idx]
                mask = target["masks"][idx]
                image, scl_box, scl_mask = self.scale_tooth(image, box, mask)

                # Update targets
                target["boxes"][idx] = scl_box
                target["masks"][idx] = scl_mask
                break

        return image, target

    def scale_tooth(self, image, box, mask):
        # Get random scale
        pct = np.random.uniform(-self.max_pct, self.max_pct)
        # Scale factor
        fs = 1 + pct / 100

        # Operate only on the first channel (grayscale 3-channel)
        h, w = np.shape(image)[:2]
        image = image[..., 0]

        # Dilate the mask
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Get the tooth and erase it from the image
        tooth = np.zeros_like(image)
        tooth[mask > 0] = np.copy(image[mask > 0])
        image[mask > 0] = 0

        # Inpaint the gap
        image = inpaint_biharmonic(image, mask)
        image = image.astype(np.float32)

        # Get the center of the original bounding box
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        x_center = round(x1 + 0.5 * box_w)
        y_center = round(y1 + 0.5 * box_h)

        # Get only tooth image from dilated mask
        dil_box = box_from_mask(mask)
        dil_x1, dil_y1, dil_x2, dil_y2 = np.round(dil_box).astype(int)
        mini_tooth = tooth[dil_y1:dil_y2, dil_x1:dil_x2]
        mini_mask = mask[dil_y1:dil_y2, dil_x1:dil_x2]

        # Scale tooth and mask
        mini_tooth = cv2.resize(mini_tooth, (0, 0), fx=fs, fy=fs)
        mini_mask = cv2.resize(mini_mask.astype(np.float32), (0, 0), fx=fs, fy=fs)

        # Erode the mask
        mini_mask = (mini_mask > 0.5).astype(np.uint8)
        # Pad 1 pixel before eroding
        mini_mask = np.pad(mini_mask, ((1, 1), (1, 1)))
        mini_mask = cv2.erode(mini_mask, kernel, iterations=1)
        # Remove extra pixel
        mini_mask = mini_mask[1:-1, 1:-1]

        # Recover the original dimensions
        scl_tooth = np.zeros_like(image)
        scl_mask = np.zeros_like(mask)
        out_h, out_w = mini_mask.shape[:2]
        y1 = y_center - out_h // 2
        x1 = x_center - out_w // 2
        y2 = y1 + out_h
        x2 = x1 + out_w

        # Cut mini tooth or mask where it goes out of image bounds
        h1 = -y1 if y1 < 0 else 0
        w1 = -x1 if x1 < 0 else 0
        h2 = out_h + min(h - y2, 0)
        w2 = out_w + min(w - x2, 0)
        mini_tooth = mini_tooth[h1:h2, w1:w2]
        mini_mask = mini_mask[h1:h2, w1:w2]

        # Introduce mini mask and tooth into image
        y1 = max(y1, 0)
        x1 = max(x1, 0)
        scl_tooth[y1:y2, x1:x2] = mini_tooth
        scl_mask[y1:y2, x1:x2] = mini_mask

        # Add the tooth to the image
        image[scl_mask > 0] = scl_tooth[scl_mask > 0]
        image = np.repeat(image[..., None], repeats=3, axis=-1)

        # Get new box from altered mask
        scl_box = box_from_mask(scl_mask)

        return image, scl_box, scl_mask
