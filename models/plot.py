"""
Visualization utilities.
"""

import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap

from .utils import box_from_mask


def plot_sample(image, labels, masks, boxes=None, tooth_labels=None, cls=None, ax=None):
    """
    Plot an image and all its masks. The color pallete
    for the masks is reduced to the number of classes of this
    project (6 colors).
    """
    # Color pallete in terms of label
    colors = {
        1: "#feb300",
        2: "#00aeae",
        3: "#007900",
        4: "#ffff00",
        5: "#ff0000",
        6: "#cc00af",
    }

    # Tooth IDs
    tooth_ids = np.array(
        [10 * i + j for i, j in itertools.product(range(1, 5), range(1, 9))]
    )
    tooth_ids = {i + 1: tooth_id for i, tooth_id in enumerate(tooth_ids)}

    # Ensure all inputs are numpy arrays
    if isinstance(image, torch.Tensor):
        image = torch.permute(image, (1, 2, 0))
    image = np.asarray(image)
    labels = np.asarray(labels)
    masks = np.asarray(masks)
    if boxes is not None:
        boxes = np.asarray(boxes)
    if tooth_labels is not None:
        tooth_labels = np.asarray(tooth_labels)

    # Ensure binary mask
    masks = (masks > 0).astype(np.uint8)

    # Express image as np.uint8 3-channel
    if image.dtype == np.float32:
        image = (255 * image).astype(np.uint8)

    # Create the plot
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(image)
    for i, mask in enumerate(masks):
        if np.any(mask > 0):
            # Get RGB color
            label = labels[i]
            if cls is not None and label != cls:
                continue
            if boxes is None:
                x1, y1, x2, y2 = box_from_mask(mask)
            else:
                x1, y1, x2, y2 = boxes[i]
                h = y2 - y1
                w = x2 - x1
                rect = patches.Rectangle(
                    (x1, y1), w, h, linewidth=1, edgecolor="r", facecolor="none"
                )
                ax.add_patch(rect)
            if label in (1, 2) and tooth_labels is not None:
                tooth_id = tooth_ids[tooth_labels[i]] if tooth_labels[i] != 0 else 0
                x_center = round((x1 + x2) / 2)
                y_center = round((y1 + y2) / 2)
                ax.text(
                    x_center,
                    y_center,
                    str(tooth_id),
                    fontsize=8,
                    color="white",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )
            alpha = (mask * 0.5) / np.max(mask)
            ax.imshow(mask, alpha=alpha, cmap=ListedColormap(["black", colors[label]]))
        else:
            continue
    ax.axis("off")
    plt.tight_layout()
