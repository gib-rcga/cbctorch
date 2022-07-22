"""
K-fold cross validation training script.
"""

import torch
import argparse
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from models.data import CBCTDataset
from models.mrcnn import get_model
from models.utils import collate_fn, Trainer
from models.transforms import *


class KFoldCrossVal:
    """
    Performs k-fold cross validation training on the CBCT dataset.
    """

    def __init__(
        self,
        data_path,
        batch_size,
        lr,
        epochs,
        accum_iters=1,
        k=4,
        model_type=0,
        augment=False,
        schedule=True,
    ):
        # Set the attributes
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.accum_iters = accum_iters
        self.k = k
        self.schedule = schedule

        self.milestones = None
        if schedule:
            if augment:
                self.milestones = [0.4 * epochs, 0.6 * epochs]
            else:
                self.milestones = [epochs / 4, epochs / 2]
            self.milestones = [round(m) for m in self.milestones]

        # Not augmented dataset
        self.dataset = CBCTDataset(data_path, transforms=ToTensor())

        # Transforms
        transforms = []
        if augment:
            transforms += [
                RandomRot(),
                RandomHFlip(self.dataset.tooth_ids),
                RandomRotDenture(),
                RandomScaleDenture(),
            ]
        transforms.append(ToTensor())
        # Concatenate transforms
        transforms = Compose(transforms)

        # Construct an augmented dataset
        self.aug_dataset = CBCTDataset(data_path, transforms=transforms)

        # Model type
        assert model_type in (0, 1)
        if model_type == 0:
            self.model_type = "full"
        elif model_type == 1:
            self.model_type = "segm"

        # Get the run name
        now = datetime.now()
        self.run_name = now.strftime("%d%m%Y_%H%M%S")

        # Obtain stratified folds
        assert self.aug_dataset.img_ids == self.dataset.img_ids  # Sanity check
        self.folds = self.dataset.strat_kfold(k=self.k)

    def fit(self):
        # Loop over the k folds
        for i in range(self.k):
            # Get the training and validation indices
            train_idxs = np.concatenate(
                [fold for j, fold in enumerate(self.folds) if j != i], axis=0
            )
            val_idxs = np.array(self.folds[i])

            # Train fold
            self.train_fold(train_idxs, val_idxs, fold=i)

    def train_fold(self, train_idxs, val_idxs, fold):
        # Initialize model
        model = get_model(dataset=self.dataset, model_type=self.model_type)

        # Create training and validation splits
        train_dataset = Subset(self.aug_dataset, train_idxs)
        val_dataset = Subset(self.dataset, val_idxs)

        # Create data loaders
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

        # Optimizer
        optimizer = torch.optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

        # LR scheduler
        if self.schedule:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.milestones,
                gamma=0.2,
            )
        else:
            lr_scheduler = None

        # Training handler
        trainer = Trainer(
            model,
            optimizer,
            epochs=self.epochs,
            train_data_loader=train_data_loader,
            accum_iters=self.accum_iters,
            val_data_loader=val_data_loader,
            run_name=self.run_name,
            fold=fold,
            lr_scheduler=lr_scheduler,
            scheduler_type="epoch",
            save_eval=True,
        )
        trainer.fit()


if __name__ == "__main__":
    # Parse input
    parser = argparse.ArgumentParser(
        description="Mask R-CNN k-fold cross validation", add_help=True
    )
    parser.add_argument(
        "--data-path", default="cbct_dataset", type=str, help="dataset path"
    )
    parser.add_argument(
        "-m",
        "--model-type",
        default=0,
        type=int,
        help="wether to train a full (0) or baseline segmentation (1) model",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2,
        type=int,
        help="images simultaneously passed to the GPU",
    )
    parser.add_argument(
        "--lr",
        default=5e-3,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="training epochs",
    )
    parser.add_argument(
        "-a",
        "--augment",
        default=False,
        action="store_true",
        help="use data augmentation",
    )
    parser.add_argument(
        "-i",
        "--accum-iters",
        default=4,
        type=int,
        help="gradient accumulation iterations",
    )
    args = parser.parse_args()

    kfold_trainer = KFoldCrossVal(
        args.data_path,
        args.batch_size,
        args.lr,
        epochs=args.epochs,
        accum_iters=args.accum_iters,
        k=4,
        model_type=args.model_type,
        augment=args.augment,
        schedule=True,
    )
    kfold_trainer.fit()
