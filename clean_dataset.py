"""
Clean the original CBCT dataset and save it as a image directory
plus a json annotation file.
"""

import os
import json
import argparse
from tqdm import tqdm
from shutil import copyfile
from pathlib import Path


def copy_images(root_dir, out_dir):
    """
    Copy the images in the database from the root_dir to
    their destination directory
    """
    # Create output directory for images
    image_dir = os.path.join(out_dir, "images")
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    # Get a list of the image file names
    images = [name for name in os.listdir(root_dir) if name.endswith(".png")]

    # Iterate over the file names and copy
    print("Copying images to destination directory...")
    for image in tqdm(images):
        copyfile(os.path.join(root_dir, image), os.path.join(image_dir, image))


def main(root_dir, json_name, out_dir):
    """
    Preprocess the dataset and save it as:
    /out_dir/images/*.png: <images in the dataset>
    /out_dir/annotations.json
    """
    # Create output dir and get input json path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(root_dir, json_name)

    # Load the json dataset
    json_dataset = json.load(open(json_path, "r"))

    # Create the three data tables (images, annotations and categories)
    imgs = {img["id"]: img for img in json_dataset["images"]}
    cats = {cat["id"]: cat for cat in json_dataset["categories"]}
    anns = {ann["id"]: ann for ann in json_dataset["annotations"]}

    # Create tooth_id field for annotations, if NaN then it has no tooth
    for ann in anns.values():
        label = cats[ann["category_id"]]["name"]
        if label == "Tooth" or label == "Metal":
            ann["tooth_id"] = ann["metadata"]["t_id"]
        else:
            ann["tooth_id"] = float("nan")

    # Keys to keep from each dataset
    img_keys = ["id", "dataset_id", "width", "height", "file_name"]
    cat_keys = ["id", "name"]
    ann_keys = ["id", "image_id", "category_id", "segmentation", "bbox", "tooth_id"]

    # Clean the dictionaries
    imgs = {img_id: {k: img[k] for k in img_keys} for img_id, img in imgs.items()}
    cats = {cat_id: {k: cat[k] for k in cat_keys} for cat_id, cat in cats.items()}
    anns = {ann_id: {k: ann[k] for k in ann_keys} for ann_id, ann in anns.items()}

    # Transform the dictionaries to dump into json
    json_dict = {
        "images": list(imgs.values()),
        "categories": list(cats.values()),
        "annotations": list(anns.values()),
    }

    # Copy the images
    copy_images(root_dir, out_dir)

    # Dump the clean dictionaries onto the output json
    json_out = os.path.join(out_dir, "annotations.json")
    with open(json_out, "w") as fp:
        json.dump(json_dict, fp)


if __name__ == "__main__":
    # Parse input
    parser = argparse.ArgumentParser(
        description="Mask R-CNN k-fold cross validation", add_help=True
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        default="./CBCT Anotado/20210415",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "-j",
        "--json-name",
        default="TrainTest-8.json",
        type=str,
        help="json name with its extension",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./cbct_dataset",
        type=str,
        help="output dataset path",
    )
    args = parser.parse_args()

    main(args.root_dir, args.json_name, args.output_dir)
