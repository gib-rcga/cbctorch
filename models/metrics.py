"""
Evaluation metrics.
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from .utils import decode_mask, compute_iou, label_nms
from .data import CBCTDataset
from .plot import plot_sample


def dice_coefficient(mask_true, mask_pred):
    """
    Compute the Sorensen-Dice coefficient from two binary masks.
    """
    # Ensure binary masks
    mask_true = (mask_true > 0).astype(np.uint8)
    mask_pred = (mask_pred > 0).astype(np.uint8)

    # Get intersection or true positives
    tp = np.sum(mask_true * mask_pred)

    return 2 * tp / (np.sum(mask_true) + np.sum(mask_pred))


def hausdorff_distance(mask_true, mask_pred):
    """
    Compute the Haussdorf distance between two masks.
    """
    # Find the contour points for the masks
    pts_1 = cv2.findContours(mask_true, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    pts_1 = np.squeeze(np.concatenate(pts_1, axis=0))
    pts_2 = cv2.findContours(mask_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    pts_2 = np.squeeze(np.concatenate(pts_2, axis=0))

    # Compute the difference between all points in both sets
    x_1, y_1 = np.split(pts_1, 2, axis=-1)
    x_2, y_2 = np.split(pts_2, 2, axis=-1)
    diff_x = x_1 - np.transpose(x_2)
    diff_y = y_1 - np.transpose(y_2)

    # Obtain the matrix of Euclidean distances
    dist = np.sqrt(diff_x**2 + diff_y**2)

    # Get the minimum distance of each point in both sets
    min_1 = np.min(dist, axis=1)
    min_2 = np.min(dist, axis=0)

    # Get the maximum between the concatenation of both distance arrays
    return np.max(np.concatenate([min_1, min_2], axis=0))


def surface_distance(mask_true, mask_pred):
    """
    Compute the average surface symmetric distance between two masks.
    """
    # Find the contour points for the masks
    pts_1 = cv2.findContours(mask_true, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    pts_1 = np.squeeze(np.concatenate(pts_1, axis=0))
    pts_2 = cv2.findContours(mask_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    pts_2 = np.squeeze(np.concatenate(pts_2, axis=0))

    # Compute the difference between all points in both sets
    x_1, y_1 = np.split(pts_1, 2, axis=-1)
    x_2, y_2 = np.split(pts_2, 2, axis=-1)
    diff_x = x_1 - np.transpose(x_2)
    diff_y = y_1 - np.transpose(y_2)

    # Obtain the matrix of Euclidean distances
    dist = np.sqrt(diff_x**2 + diff_y**2)

    # Get the minimum distance of each point in both sets
    min_1 = np.min(dist, axis=1)
    min_2 = np.min(dist, axis=0)

    # Return the average surface symmetric distance
    return (np.sum(min_1) + np.sum(min_2)) / (len(min_1) + len(min_2))


class Evaluator:
    """
    Computes the evaluation metrics from ground truths and detections:
    - COCO mAP
    - Dice coefficient
    - Hausdorff distance
    - Average symmetric surface distance
    - Tooth classification accuracy
    """

    def __init__(self, dataset, json_path, res=0.16, nms=True, ap_label=None):
        # Read the evaluation data
        evaluation_data = json.load(open(json_path, "r"))

        # Object labels
        assert isinstance(dataset, CBCTDataset)
        self.dataset = dataset
        self.cat_names = dataset.cat_names
        self.tooth_ids = dataset.tooth_ids

        # Pixel resolution in mm
        self.res = res

        # Get the prepared ground truths and detections
        self.gts = self._prepare_dict(evaluation_data["targets"])
        self.dts = self._prepare_dict(evaluation_data["detections"])

        # NMS detections
        if nms:
            nms_idxs = {
                i: label_nms(d["boxes"], d["scores"], d["labels"], iou_thresh=0.5)
                for i, d in self.dts.items()
            }
            self.dts = {
                i: {k: v[nms_idxs[i]] for k, v in d.items()}
                for i, d in self.dts.items()
            }

        # Sort detections by score
        dts_idxs = {i: np.argsort(d["scores"])[::-1] for i, d in self.dts.items()}
        self.dts = {
            i: {k: v[dts_idxs[i]] for k, v in d.items()} for i, d in self.dts.items()
        }

        # Compute the matches of each detection and prediction
        # Indexes by image ID and IoU thresholds @[0.5:0.95:0.05]
        self.gt_matches, self.dt_matches = {}, {}
        for img_id in self.gts.keys():
            if ap_label is not None:
                gt_idxs = np.argwhere(self.gts[img_id]["labels"] == ap_label)[:,0]
                if len(gt_idxs) == 0:
                    continue
            gt_m, dt_m = self._compute_matches(self.gts[img_id], self.dts[img_id], label=ap_label)
            self.gt_matches[img_id] = gt_m
            self.dt_matches[img_id] = dt_m

    def plot_sample(self, idx, plot_gt=True, plot_dt=True, cls=None, boxes=False, tooth_labels=False):
        """
        Plots the ground truth and prediction of an image.
        """
        assert idx < len(self.dataset.img_ids)
        img_id = self.dataset.img_ids[idx]
        image, _ = self.dataset[idx]
        gt = self.gts[img_id]
        dt = self.dts[img_id]
        gt_args = {"labels": gt["labels"], "masks": gt["masks"]}
        dt_args = {"labels": dt["labels"], "masks": dt["masks"]}
        if boxes:
            gt_args["boxes"] = gt["boxes"]
            dt_args["boxes"] = dt["boxes"]
        if tooth_labels:
            gt_args["tooth_labels"] = gt["tooth_labels"]
            dt_args["tooth_labels"] = dt["tooth_labels"]

        # Create subplot
        if plot_gt and plot_dt:
            _, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
            plot_sample(image, **gt_args, cls=cls, ax=axs[0])
            plot_sample(image, **dt_args, cls=cls, ax=axs[1])
        elif plot_gt:
            _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
            plot_sample(image, **gt_args, cls=cls, ax=ax)
        elif plot_dt:
            _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
            plot_sample(image, **dt_args, cls=cls, ax=ax)

    def eval(self, out_dir="./", excel_name=None, teeth=True):
        """
        Performs all evaluations.
        """
        # Construct result
        result = {}
        result["ap"] = self.eval_ap()
        result["dsc"] = self.eval_dsc()
        result["hd"] = self.eval_hd()
        result["assd"] = self.eval_assd()
        result["mat1"] = self.get_conf_mat()

        if teeth:
            result["acc"] = self.eval_acc()
            prc, rec = self.eval_pc()
            result["prc"] = prc
            result["rec"] = rec
            result["mat2"] = self.get_conf_mat(key="tooth_labels")

        # Save as Excel
        if excel_name is not None:
            result_df = {
                k: pd.DataFrame(v).round(2) for k, v in result.items() if "mat" not in k
            }
            metric_names = {
                "ap": "Mean AP",
                "dsc": "Dice Coefficient",
                "hd": "Hausdorff Distance",
                "assd": "ASSD",
                "acc": "Accuracy",
                "prc": "Precision",
                "rec": "Recall",
            }
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            writer = pd.ExcelWriter(
                os.path.join(out_dir, excel_name), engine="xlsxwriter"
            )
            for k, df in result_df.items():
                df.to_excel(writer, sheet_name=metric_names[k])
            writer.save()

        return result

    def eval_ap(self):
        """
        Obtains the mAP of the model at IoU range [0.5:1.0:0.05], threshold
        0.5 and threshold 0.75.
        """
        # COCO IoU range
        iou_range = np.round(np.arange(0.5, 1, 0.05), 2)
        # Result mAP
        ap = {
            "AP": [],
            "AP50": [],
            "AP75": [],
            "AP60": [],
            "AP70": [],
            "AP80": [],
            "AP90": [],
        }

        # Iterate over all images
        for img_id in self.gt_matches.keys():
            ap_range = {}
            for t in iou_range:
                ap_range[t] = self.compute_ap(img_id, t)
            ap["AP"].append(np.mean(list(ap_range.values())))
            ap["AP50"].append(ap_range[0.5])
            ap["AP75"].append(ap_range[0.75])
            ap["AP60"].append(ap_range[0.6])
            ap["AP70"].append(ap_range[0.7])
            ap["AP80"].append(ap_range[0.8])
            ap["AP90"].append(ap_range[0.9])

        # Compute the mean and std of each mAP
        return {
            k: {
                "mean": np.mean(v) * 100,
                "std": np.std(v) * 100,
            }
            for k, v in ap.items()
        }

    def eval_pc(self):
        """
        Evaluate the tooth classifier precision and recall.
        """
        # All predictions and ground truths
        tooth_pred = []
        tooth_true = []

        # Iterate over the detections with threshold 0.5
        for img_id in self.gt_matches.keys():
            # Get the indices for predictions and detections
            gt_matches = self.gt_matches[img_id][0.5]
            # Loop over matches
            for gt, dt in enumerate(gt_matches):
                if dt > -1:
                    tooth_true.append(self.gts[img_id]["tooth_labels"][gt])
                    tooth_pred.append(self.dts[img_id]["tooth_labels"][dt])

        # Express as numpy arrays
        tooth_pred = np.array(tooth_pred)
        tooth_true = np.array(tooth_true)

        # Precision and recall dicts
        prc = []
        rec = []

        # Iterate over teeth compute the precision and recall
        for label in self.tooth_ids.values():
            # Compute TP, FP and FN
            tp = self.comp_tp(tooth_true, tooth_pred, label)
            fp = self.comp_fp(tooth_true, tooth_pred, label)
            fn = self.comp_fn(tooth_true, tooth_pred, label)

            # Compute precision and recall
            if tp > 0:
                prc.append((tp / (fp + tp)) * 100)
                rec.append((tp / (fn + tp)) * 100)

        prc = {"Tooth": {"mean": np.mean(prc), "std": np.std(prc)}}
        rec = {"Tooth": {"mean": np.mean(rec), "std": np.std(rec)}}

        return prc, rec

    @staticmethod
    def comp_tp(true, pred, label):
        # Get the gts equal to label
        label_idxs = np.argwhere(true == label)[:, 0]
        gt = true[label_idxs]
        yp = pred[label_idxs]

        return np.sum(gt == yp)

    @staticmethod
    def comp_fp(true, pred, label):
        # Get the detections equal to label
        label_idxs = np.argwhere(pred == label)[:, 0]
        gt = true[label_idxs]
        yp = pred[label_idxs]

        return np.sum(gt != yp)

    @staticmethod
    def comp_fn(true, pred, label):
        # Get the gts equal to label
        label_idxs = np.argwhere(true == label)[:, 0]
        gt = true[label_idxs]
        yp = pred[label_idxs]

        return np.sum(gt != yp)

    def eval_acc(self, missing_teeth=None):
        """
        Evaluate the tooth classification accuracy.
        """
        # Per image accuracy
        acc = []
        tooth_label = {v: k for k, v in self.tooth_ids.items()}

        # Iterate over the detections with threshold 0.5
        for img_id in self.gt_matches.keys():
            matches = []
            # Get the indices for predictions and detections
            gt_matches = self.gt_matches[img_id][0.5]
            # Loop over matches
            for gt, dt in enumerate(gt_matches):
                if dt > -1:
                    tooth_true = self.gts[img_id]["tooth_labels"][gt]
                    # Skip prediction if tooth is in missing teeth
                    if missing_teeth is not None and tooth_true > 0:
                        if tooth_label[tooth_true] in missing_teeth:
                            continue
                    tooth_pred = self.dts[img_id]["tooth_labels"][dt]
                    if tooth_true > 0:
                        matches.append(tooth_true == tooth_pred)
            if len(matches) > 0:
                acc.append(np.sum(matches) / len(matches))

        return {
            "Tooth": {
                "mean": np.mean(acc) * 100,
                "std": np.std(acc) * 100,
            }
        }

    def get_conf_mat(self, key="labels", missing_teeth=None):
        """
        Obtains the confusion matrix of the tooth classifier.
        """
        # Create the confusion matrix
        if key == "labels":
            n = len(self.cat_names) + 1  # Account for background
        else:
            n = len(self.tooth_ids) + 1  # Account for background
        mat = np.zeros((n, n), dtype=int)
        tooth_label = {v: k for k, v in self.tooth_ids.items()}

        # Iterate over the detections with threshold 0.5
        for img_id in self.gt_matches.keys():
            # Get the indices for predictions and detections
            gt_matches = self.gt_matches[img_id][0.5]
            # Loop over matches
            for gt, dt in enumerate(gt_matches):
                if dt > -1:
                    label_true = self.gts[img_id][key][gt]
                    # Skip prediction if tooth is in missing teeth
                    if missing_teeth is not None and label_true > 0:
                        if tooth_label[label_true] in missing_teeth:
                            continue
                    label_pred = self.dts[img_id][key][dt]
                    # Account for tooth labels belonging to
                    mat[label_true, label_pred] += 1

        return mat

    def eval_dsc(self):
        """
        Get the mean Dice coefficient of all predicted masks.
        """
        # Initialize per-class DSC
        dsc = defaultdict(list)

        # Iterate over the detections with threshold 0.5
        for img_id in self.gt_matches.keys():
            # Get the indices for predictions and detections
            gt_matches = self.gt_matches[img_id][0.5]
            # Loop over matches
            for gt, dt in enumerate(gt_matches):
                if dt > -1:
                    label = self.gts[img_id]["labels"][gt]
                    mask_true = self.gts[img_id]["masks"][gt]
                    mask_pred = self.dts[img_id]["masks"][dt]
                    dsc[label].append(dice_coefficient(mask_true, mask_pred))

        # Get the mean and standard deviation Dice coefficient
        return {
            self.cat_names[k]: {
                "mean": np.mean(dsc[k]),
                "std": np.std(dsc[k]),
            }
            for k in sorted(dsc.keys())
        }

    def eval_assd(self):
        """
        Obtain the average symmetric surface distance.
        """
        assd = defaultdict(list)

        # Iterate over the detections with threshold 0.5
        for img_id in self.gt_matches.keys():
            # Get the indices for predictions and detections
            gt_matches = self.gt_matches[img_id][0.5]
            # Loop over matches
            for gt, dt in enumerate(gt_matches):
                if dt > -1:
                    label = self.gts[img_id]["labels"][gt]
                    mask_true = self.gts[img_id]["masks"][gt]
                    mask_pred = self.dts[img_id]["masks"][dt]
                    if np.any(mask_true) and np.any(mask_pred):
                        assd[label].append(
                            self.res * surface_distance(mask_true, mask_pred)
                        )

        # Get the mean and standard deviation ASSD
        return {
            self.cat_names[k]: {
                "mean": np.mean(assd[k]),
                "std": np.std(assd[k]),
            }
            for k in sorted(assd.keys())
        }

    def eval_hd(self):
        """
        Get the Hausdorff distances of the ground truth and predicted masks.
        """
        # Initialize per-class HD
        hd = defaultdict(list)

        # Iterate over the detections with threshold 0.5
        for img_id in self.gt_matches.keys():
            # Get the indices for predictions and detections
            gt_matches = self.gt_matches[img_id][0.5]
            # Loop over matches
            for gt, dt in enumerate(gt_matches):
                if dt > -1:
                    label = self.gts[img_id]["labels"][gt]
                    mask_true = self.gts[img_id]["masks"][gt]
                    mask_pred = self.dts[img_id]["masks"][dt]
                    if np.any(mask_true) and np.any(mask_pred):
                        hd[label].append(
                            self.res * hausdorff_distance(mask_true, mask_pred)
                        )

        # Get the mean and standard deviation Dice coefficient
        return {
            self.cat_names[k]: {
                "mean": np.mean(hd[k]),
                "std": np.std(hd[k]),
            }
            for k in sorted(hd.keys())
        }

    def compute_ap(self, img_id, iou_thresh):
        """
        Computes the mAP of an image at certain threshold.
        """
        # Retrieve the matches
        gt_matches = self.gt_matches[img_id][iou_thresh]
        dt_matches = self.dt_matches[img_id][iou_thresh]

        # Compute precision and recall
        p = np.cumsum(dt_matches > -1) / (np.arange(len(dt_matches)) + 1)
        r = np.cumsum(dt_matches > -1) / len(gt_matches)

        # Pad precisions and recalls
        p = np.concatenate([[0], p, [0]])
        r = np.concatenate([[0], r, [1]])

        # Round the precisions to their rightmost maximum
        p = np.maximum.accumulate(p[::-1])[::-1]
        # Get the recall indices over a set levels (101)
        r_vals = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        r_idxs = np.searchsorted(r, r_vals, side="left")

        # Get the interpolated values of the precision in those levels
        p_interp = np.array([p[r_idx] if r_idx < len(p) else 0 for r_idx in r_idxs])

        # Return mean AP
        return np.mean(p_interp)

    def _prepare_dict(self, data):
        # Expects List[Dict], and indexes them by image ID
        prep = {
            d["image_id"]: {k: v for k, v in d.items() if k != "image_id"} for d in data
        }

        # Decode the RLE masks
        for d in prep.values():
            d["masks"] = [decode_mask(r) for r in d["masks"]]

        # Espress all data as numpy arrays
        prep = {i: {k: np.array(v) for k, v in d.items()} for i, d in prep.items()}

        return prep

    def _compute_matches(self, gt, dt, label=None):
        # COCO IoU range
        iou_range = np.round(np.arange(0.5, 1, 0.05), 2)

        # Unpack the ground truths
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]

        # If given a label, only compute AP for that label
        if label is not None:
            gt_idxs = np.argwhere(gt_labels == label)[:,0]
            gt_boxes = gt_boxes[gt_idxs]
            gt_labels = gt_labels[gt_idxs]

        # Unpack detections (already sorted in descending order of score)
        dt_boxes = dt["boxes"]
        dt_labels = dt["labels"]

        # Compute the IoU between both boxes
        iou = compute_iou(dt_boxes, gt_boxes)
        gt_matches = {
            t: -1 * np.ones(len(gt_labels), dtype=np.int64) for t in iou_range
        }
        dt_matches = {
            t: -1 * np.ones(len(dt_labels), dtype=np.int64) for t in iou_range
        }
        # Loop over the thresholds
        for t in iou_range:
            # Loop over detections
            for i in range(len(dt_labels)):
                # Sort IoUs of detection in descending order
                candidate_gts = np.argsort(iou[i])[::-1]
                # Iterate over the candidates
                for j in candidate_gts:
                    # Check if gt is already matched, skip
                    if gt_matches[t][j] > -1:
                        continue
                    # If the IoU is lower than threshold go to next detection
                    if iou[i, j] < t:
                        break
                    # If the labels match, add it to matches
                    if dt_labels[i] == gt_labels[j]:
                        gt_matches[t][j] = i
                        dt_matches[t][i] = j

        return gt_matches, dt_matches


class FoldEvaluator:
    """
    Wrapper of Evaluator objects for k-fold cross validation.
    """

    def __init__(self, dataset, log_dir, save_dir, teeth=True, ap_label=None):
        # Directories
        folds = [os.path.join(log_dir, d) for d in os.listdir(log_dir)]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.teeth = teeth

        # Evaluators
        json_paths = [os.path.join(d, "evaluation.json") for d in folds]
        self.evals = [Evaluator(dataset, p, ap_label=ap_label) for p in json_paths]

        # Results
        self.results = None
        self.results_df = None

    def get_results(self):
        # Get all the fold results
        for e in self.evals:
            result = e.eval(teeth=self.teeth)
            if self.results is None:
                self.results = {
                    m: {k: {"mean": [], "std": []} for k in d.keys()}
                    for m, d in result.items()
                    if "mat" not in m
                }
            for m, d in result.items():
                if "mat" not in m:
                    for k in d.keys():
                        self.results[m][k]["mean"].append(result[m][k]["mean"])
                        self.results[m][k]["std"].append(result[m][k]["std"])

        for m, d in self.results.items():
            for k in d.keys():
                self.results[m][k]["mean"] = np.mean(self.results[m][k]["mean"])
                self.results[m][k]["std"] = np.sqrt(
                    np.sum(np.power(self.results[m][k]["std"], 2))
                ) / len(self.results[m][k]["std"])

        result_df = {
            k: pd.DataFrame(v).round(2)
            for k, v in self.results.items()
            if "mat" not in k
        }
        metric_names = {
            "ap": "Mean AP",
            "dsc": "Dice Coefficient",
            "hd": "Hausdorff Distance",
            "assd": "ASSD",
            "acc": "Accuracy",
            "prc": "Precision",
            "rec": "Recall",
        }
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        writer = pd.ExcelWriter(
            os.path.join(self.save_dir, "kfold_results.xlsx"), engine="xlsxwriter"
        )
        for k, df in result_df.items():
            df.to_excel(writer, sheet_name=metric_names[k])
        writer.save()
