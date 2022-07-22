"""
Dataset utilities.
"""
import os
import cv2
import json
import random
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset

from .utils import fill_mask


class CBCTDataset(Dataset):
    """
    Torch Dataset handling the CBCT data. The expected
    structure of the data directory is:
    data_dir/
        images/
        annotations.json
    """

    def __init__(
        self,
        data_dir,
        transforms=None,
        active_labels=["Tooth", "Metal", "Maxilla", "Mandible", "Maxillary Sinus"],
        only_teeth=False,
    ):
        # Get the image directory
        self.img_dir = os.path.join(data_dir, "images")
        # Data transforms
        self.transforms = transforms

        # Read the json; index the dataset
        annotation_file = os.path.join(data_dir, "annotations.json")
        json_dataset = json.load(open(annotation_file, "r"))

        # Set the active labels
        self._active_labels = active_labels
        if only_teeth:
            self._active_labels = ["Tooth", "Metal"]

        # Get active category IDs and tooth IDs
        self._set_categories(json_dataset)
        self._set_tooth_ids()

        # Obtain the images, their annotations and categories
        self.imgs = {img["id"]: img for img in json_dataset["images"]}
        self.cats = {cat["id"]: cat for cat in json_dataset["categories"]}
        self.img_anns = defaultdict(list)
        self.aug_anns = defaultdict(list)  # All annotations for use in augmentation
        for ann in json_dataset["annotations"]:
            # Append annotation only if it belongs to an active label
            cat_id = ann["category_id"]
            if cat_id in self.cat_labels.keys():
                self.img_anns[ann["image_id"]].append(ann)
            self.aug_anns[ann["image_id"]].append(ann)

        # Get only image IDs with at least one annotation
        self.img_ids = list(self.img_anns.keys())

        # Get the teeth not in this dataset (only seen as metal)
        self._set_missing_teeth()

        # Get the minimum and maximum dimensions in the dataset
        img_dims = [
            [self.imgs[img_id]["height"], self.imgs[img_id]["width"]]
            for img_id in self.img_ids
        ]
        self.min_size = np.min(img_dims).astype(np.int64)
        self.max_size = np.max(img_dims).astype(np.int64)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get the image ID from the index
        img_id = self.img_ids[idx]

        # Read the image
        image = self.get_image(img_id)

        # Get the labels, boxes and masks
        anns = self.img_anns[img_id]
        target = {}
        target["boxes"] = [self.get_box(ann) for ann in anns]
        target["labels"] = [self.cat_labels[ann["category_id"]] for ann in anns]
        target["tooth_labels"] = [self.get_tooth(ann) for ann in anns]
        target["masks"] = [self.get_mask(ann) for ann in anns]
        target["image_id"] = img_id

        # Express targets as numpy arrays
        target = {k: np.array(v) for k, v in target.items()}

        # Ensure data types
        dtypes = {
            "boxes": np.float32,
            "labels": np.int64,
            "tooth_labels": np.int64,
            "masks": np.uint8,
            "image_id": np.int64,
        }
        for k in target.keys():
            target[k] = target[k].astype(dtypes[k])

        # Apply transforms or augmentations
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _set_categories(self, json_dataset):
        """
        Prepares the category IDS, names and the number of classes.
        """
        # Get active category IDs by name
        if len(self._active_labels) != 0:
            cats = [
                cat
                for cat in json_dataset["categories"]
                if cat["name"] in self._active_labels
            ]
        else:
            # If active labels is empty, get all categories of the dataset
            cats = json_dataset["categories"]
        active_ids = [cat["id"] for cat in cats]  # IDs of active labels

        # Map them to labes [1, len(active_labels)]
        self.cat_labels = {cat_id: i + 1 for i, cat_id in enumerate(active_ids)}
        self.cat_names = {i + 1: cat["name"] for i, cat in enumerate(cats)}
        self.num_classes = len(self.cat_labels) + 1  # Number of classes plus background

    def _set_tooth_ids(self):
        """
        Expresses FDI notation IDs as [1, num_teeth] labels.
        """
        # Express the tooth IDs (FDI notation) as [1, num_teeth]
        tooth_ids = np.array(
            [10 * i + j for i, j in itertools.product(range(1, 5), range(1, 9))]
        )
        self.tooth_ids = {tooth_id: i + 1 for i, tooth_id in enumerate(tooth_ids)}

    def _set_missing_teeth(self):
        # Get all the teeth ids for each patient in the dataset
        teeth = []
        for img_id in self.img_ids:
            anns = self.img_anns[img_id]
            teeth.append(
                [
                    ann["tooth_id"]
                    for ann in anns
                    if self.cats[ann["category_id"]]["name"] == "Tooth"
                ]
            )

        # Concatenate and keep unique teeth
        teeth = np.unique(np.concatenate(teeth)).astype(int)

        # Missing teeth
        self.missing_teeth = np.setdiff1d(
            list(self.tooth_ids.keys()), teeth, assume_unique=True
        )

    def get_image(self, img_id):
        """
        Reads an image, returns a np.uint8 RGB array.
        """
        # Read the image
        img_path = os.path.join(self.img_dir, self.imgs[img_id]["file_name"])
        image = cv2.imread(img_path)

        # If image is grayscale convert to RGB grayscale
        assert np.ndim(image) >= 2
        if np.ndim(image) < 3:
            image = np.repeat(image[..., None], 3)

        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def get_tooth(self, ann):
        """
        Translates tooth ID to label, handling NaNs.
        """
        if np.isnan(ann["tooth_id"]):
            return 0
        return self.tooth_ids[ann["tooth_id"]]

    def get_box(self, ann):
        """
        Obtains the bounding box from an annotation.
        """
        # Obtain them as [x1, y1, x2, y2]
        x1, y1, w, h = ann["bbox"]
        x2 = x1 + w
        y2 = y1 + h

        return [x1, y1, x2, y2]

    def get_mask(self, ann):
        """
        Obtains a binary uint8 mask from an annotation.
        """
        # Get the target image height and width
        img = self.imgs[ann["image_id"]]
        h, w = img["height"], img["width"]

        # Obtain the mask from the polygon contours
        segm = ann["segmentation"]
        mask = fill_mask(segm, [h, w])

        return mask

    def get_tissue(self, img_id, tissue=0):
        """
        Obtains a single mask representing either soft (0) tissue, bone (1) tissue or metal (2).
        Implemented for NMAR augmentation.
        """
        assert tissue in (0, 1, 2)

        # Get the metal mask annotations
        anns = [
            ann
            for ann in self.aug_anns[img_id]
            if self.cats[ann["category_id"]]["name"] == "Metal"
        ]

        # Load all metal masks and do logical OR
        metal_mask = np.array([self.get_mask(ann) for ann in anns])  # [N, H, W]
        metal_mask = np.any(metal_mask, axis=0)  # Boolean array [H, W]

        # Return if metal
        if tissue == 2:
            return metal_mask.astype(np.uint8)

        # Get the bone masks
        cat_names = ["Tooth", "Maxila", "Mandible"]
        anns = [
            ann
            for ann in self.aug_anns[img_id]
            if self.cats[ann["category_id"]]["name"] in cat_names
        ]

        # Load all the masks and do a logical OR
        bone_mask = np.array([self.get_mask(ann) for ann in anns])  # [N, H, W]
        bone_mask = np.any(bone_mask, axis=0)  # Boolean array [H, W]

        # Return if bone tissue
        if tissue == 1:
            return bone_mask.astype(np.uint8)

        # Combine metal and bone as hard material mask
        hard_mask = np.logical_or(metal_mask, bone_mask)

        # If not load and threshold the image, removing the bone
        image = self.get_image(img_id)
        soft_mask = np.any(image, axis=-1)  # Logical OR accross RGB channels

        # XOR soft tissue with hard materials
        soft_mask = np.logical_xor(hard_mask, soft_mask)

        return soft_mask.astype(np.uint8)

    def box_statistics(self):
        """
        Plots the histogram of mean box sizes.
        """
        sizes = []

        # Iterate over the annotations
        anns = sum(list(self.img_anns.values()), [])
        for ann in anns:
            # Get mean size of box
            _, _, h, w = ann["bbox"]
            size = (h + w) / 2

            # Append to all sizes
            sizes.append(size)

        # Plot probability distribution
        hist, bins = np.histogram(sizes, bins=100, density=True)
        bin_centers = (bins[1:] + bins[:-1]) * 0.5
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.plot(bin_centers, hist)
        ax.grid()
        ax.set_xscale("log", base=2)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Box size")
        ax.set_ylim(ymin=0)
        locmaj = matplotlib.ticker.LogLocator(base=2, subs=(1.0,), numticks=100)
        ax.get_xaxis().set_major_locator(locmaj)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        locmin = matplotlib.ticker.LogLocator(
            base=2, subs=np.arange(2, 10) * 0.1, numticks=100
        )
        ax.get_xaxis().set_minor_locator(locmin)
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.tight_layout()

    def cat_statistics(self):
        """
        Returns a dictionary with the number of annotations per category.
        """
        cat_anns = {name: 0 for name in self.cat_names.values()}
        for anns in self.img_anns.values():
            for ann in anns:
                cat_name = self.cats[ann["category_id"]]["name"]
                cat_anns[cat_name] += 1
        
        return cat_anns

    def encode_label(self, img_id):
        # Get annotations and labels
        anns = self.img_anns[img_id]
        labels = np.array([self.cat_labels[ann["category_id"]] for ann in anns])

        # Encode labels into a binary array (multi-class) with object labels
        # (no background)
        bin_labels = np.zeros(shape=self.num_classes - 1, dtype=np.uint8)
        bin_labels[labels - 1] = 1

        # Transform to decimal number
        return bin_labels.dot(1 << np.arange(bin_labels.shape[-1]))

    def label_dist(self):
        # Get the label (multi-class) distribution
        dist = defaultdict(list)
        for idx in range(len(self.img_ids)):
            # Encoded label
            img_id = self.img_ids[idx]
            label = self.encode_label(img_id)
            dist[label].append(idx)

        # Order the dictionary
        return {k: dist[k] for k in sorted(dist.keys())}

    def strat_kfold(self, k=4):
        """
        Splits the dataset indices using stratified k-fold
        cross validation.
        """
        # Get the label distribution
        dist = self.label_dist()

        # Create the folds
        folds = [[] for _ in range(k)]

        # Iterate over the labels
        for idxs in dist.values():
            # Shuffle indices
            random.shuffle(idxs)
            # Split the indices
            splits = [idxs[i::k] for i in range(k)]
            # Shuffle splits to avoid bigger first fold
            random.shuffle(splits)

            # Append to the folds
            for fold, split in zip(folds, splits):
                fold += split

        # Shuffle each fold
        for fold in folds:
            random.shuffle(fold)

        return folds

    def summary(self):
        """
        Print a summary of the dataset.
        """
        print(f"Valid images: {len(self)}")
        print("Active label mapping:")
        for k, v in self.cat_names.items():
            print(f" - {k}: {v}")
