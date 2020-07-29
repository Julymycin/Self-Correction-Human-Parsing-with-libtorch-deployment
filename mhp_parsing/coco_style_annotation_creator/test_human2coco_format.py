import argparse
import datetime
import json
import os
from PIL import Image

from coco_style_annotation_creator import pycococreatortools



LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
    },
]


def tococo(img_dir,json_save_dir, dataname):
    INFO = {
    "description": dataname + "Dataset",
    "url": "",
    "version": "",
    "year": 2020,
    "contributor": "qiu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1

    for image_name in os.listdir(img_dir):
        image = Image.open(os.path.join(img_dir, image_name))
        image_info = pycococreatortools.create_image_info(
            image_id, image_name, image.size
        )
        coco_output["images"].append(image_info)
        image_id += 1

    if not os.path.exists(os.path.join(json_save_dir)):
        os.mkdir(os.path.join(json_save_dir))

    with open('{}/{}.json'.format(json_save_dir, dataname), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


# if __name__ == "__main__":
#     tococo()
