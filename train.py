"""
Training script.
"""

import torch
import argparse
from torch.utils.data import DataLoader

from models.data import CBCTDataset
from models.mrcnn import get_model
from models.utils import collate_fn, Trainer
from models.transforms import *


def main(
    data_path,
    batch_size,
    lr,
    epochs,
    accum_iters=1,
    model_type=0,
    augment=False,
    schedule=True,
):
    """
    Training function.
    """
    # Reference dataset
    dataset = CBCTDataset(data_path)

    milestones = None
    if schedule:
        if augment:
            milestones = [0.4 * epochs, 0.6 * epochs]
        else:
            milestones = [epochs / 4, epochs / 2]
        milestones = [round(m) for m in milestones]

    # Transforms
    transforms = []
    if augment:
        transforms += [
            RandomRot(),
            RandomHFlip(dataset.tooth_ids),
            RandomRotDenture(),
            RandomScaleDenture(),
        ]
    transforms.append(ToTensor())
    # Concatenate transforms
    transforms = Compose(transforms)

    # Construct an augmented an not augmented dataset
    aug_dataset = CBCTDataset(data_path, transforms=transforms)

    # Model type
    assert model_type in (0, 1)
    if model_type == 0:
        model_type = "full"
    elif model_type == 1:
        model_type = "segm"

    # Initialize the model
    model = get_model(dataset=dataset, model_type=model_type)

    # Create data loaders
    data_loader = DataLoader(
        aug_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    # LR scheduler
    if schedule:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.2,
        )
    else:
        lr_scheduler = None

    # Training handler
    trainer = Trainer(
        model,
        optimizer,
        epochs=epochs,
        train_data_loader=data_loader,
        accum_iters=accum_iters,
        lr_scheduler=lr_scheduler,
        scheduler_type="epoch",
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

    main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        accum_iters=args.accum_iters,
        model_type=args.model_type,
        augment=args.augment,
        schedule=True,
    )
