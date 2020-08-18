import argparse
import datetime
import json
import os
from PIL import Image
import numpy as np

import pycococreatortools


def get_arguments():
    parser = argparse.ArgumentParser(
        description="transform mask annotation to coco annotation")
    parser.add_argument("--dataset",
                        type=str,
                        default='MHPv2',
                        help="name of dataset (CIHP, MHPv2 or VIP)")
    parser.add_argument("--json_save_dir",
                        type=str,
                        default='../data/msrcnn_finetune_annotations',
                        help="path to save coco-style annotation json file")
    parser.add_argument("--use_val",
                        type=bool,
                        default=False,
                        help="use train+val set for finetuning or not")
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default='../data/instance-level_human_parsing/Training/Images',
        help="train image path")
    parser.add_argument(
        "--train_anno_dir",
        type=str,
        default='../data/instance-level_human_parsing/Training/Human_ids',
        help="train human mask path")
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default='../data/instance-level_human_parsing/Validation/Images',
        help="val image path")
    parser.add_argument(
        "--val_anno_dir",
        type=str,
        default='../data/instance-level_human_parsing/Validation/Human_ids',
        help="val human mask path")
    return parser.parse_args()


def main(args):
    INFO = {
        "description": args.split_name + " Dataset",
        "url": "",
        "version": "",
        "year": 2019,
        "contributor": "xyq",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [{"id": 1, "name": "", "url": ""}]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
        },
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    train_imgs = os.listdir(args.train_img_dir)
    for idx, image_name in enumerate(train_imgs):
        if image_name.endswith(['jpg', 'png']):
            image = Image.open(os.path.join(args.train_img_dir, image_name))
            image_info = pycococreatortools.create_image_info(
                image_id, image_name, image.size)
            coco_output["images"].append(image_info)
            mask_name_prefix = image_name.split('.')[0]
            temp_i = 0
            mask_temp = np.ones((image.size[1], image.size[0]))
            while (True):
                mask_path = os.path.join(
                    args.train_annos_dir,
                    mask_name_prefix + '_02_0' + temp_i + '.png')
                if os.path.exists(mask_path):
                    mask = np.asarray(Image.open(mask_path))
                    mask_temp = mask_temp & mask
                    temp_i += 1
                else:
                    break
            gt_labels = np.unique(mask_temp)
            for i in range(1, len(gt_labels)):
                category_info = {'id': 1, 'is_crowd': 0}
                binary_mask = np.uint8(mask_temp == i)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id,
                    image_id,
                    category_info,
                    binary_mask,
                    image.size,
                    tolerance=10)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                segmentation_id += 1
            image_id += 1
    if not os.path.exists(args.json_save_dir):
        os.makedirs(args.json_save_dir)

    if not args.use_val:
        with open(
                '{}/{}_train.json'.format(args.json_save_dir, args.split_name),
                'w') as output_json_file:
            json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
