#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import torch
import cv2

from torch.utils import data
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import sys
sys.path.append('/home/qiu/Projects/Self-Correction-Human-Parsing/')
import networks
from utils.miou import compute_mean_ioU
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing, transform_logits
from utils.transforms import get_affine_transform
from global_local_parsing.global_local_datasets import CropDataValSet


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101_3')
    # Data Preference
    parser.add_argument("--data-dir",
                        type=str,
                        default='mhp_extension/data/DemoDataset')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--split-name", type=str, default='crop_pic')
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Evaluation Preference
    parser.add_argument(
        "--log-dir",
        type=str,
    )
    parser.add_argument(
        "--model-restore",
        type=str,
        default=
        '/home/qiu/Downloads/models/detectron2/exp_schp_multi_cihp_local.pth')
    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="choose gpu device.")
    parser.add_argument("--save-results",
                        default=True,
                        action="store_true",
                        help="whether to save the results.")
    parser.add_argument("--flip",
                        action="store_true",
                        help="random flip during the test.")
    parser.add_argument("--multi-scales",
                        type=str,
                        default='1',
                        help="multiple scales during the test")
    return parser.parse_args()


def multi_scale_testing(model,
                        batch_input_im,
                        crop_size=[473, 473],
                        flip=False,
                        multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(size=crop_size,
                               mode='bilinear',
                               align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s,
                                      mode='bilinear',
                                      align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1]
        output = parsing_output[0]
        # print(output.shape)
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output.unsqueeze(0))
        ms_outputs.append(output[0])
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)
    ms_fused_parsing_output = ms_fused_parsing_output.permute(1, 2, 0)  # HWC
    parsing = torch.argmax(ms_fused_parsing_output, dim=2)
    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()
    return parsing, ms_fused_parsing_output


def glparsing(data_dir, split_name, schp_ckpt, log_dir, file_list):
    """Create the model and start the evaluation process."""
    args = get_arguments()
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True
    cudnn.enabled = True

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    model = networks.init_model(args.arch,
                                num_classes=args.num_classes,
                                pretrained=None)

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
    if INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])

    # Load model weight
    state_dict = torch.load(schp_ckpt)
    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, meta in enumerate(file_list):
            src_name = meta['im_name']
            src = cv2.imread(os.path.join(data_dir, 'src_imgs', src_name),
                             cv2.IMREAD_COLOR)
            parsing_results = []
            for i in range(meta['person_num'] + 1):
                if i == 0:
                    img = src
                else:
                    x_min, y_min, x_max, y_max = meta['person_bbox'][i - 1]
                    img = src[y_min:y_max + 1, x_min:x_max + 1, :]
                height, width, _ = img.shape
                c = np.array([height / 2, width / 2], dtype=np.float32)
                temps = max(width, height) - 1
                s = np.array([temps * 1.0, temps * 1.0], dtype=np.float32)
                r = 0
                trans = get_affine_transform(c, s, r, input_size)
                img = cv2.warpAffine(img,
                                     trans,
                                     (int(input_size[1]), int(input_size[0])),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
                img = transform(img)
                parsing, logits = multi_scale_testing(
                    model,
                    img.cuda(),
                    crop_size=input_size,
                    flip=args.flip,
                )
                parsing[parsing != 2] = 0
                parsing_result = transform_parsing(parsing, c, s, width, height,
                                                   input_size)
                parsing_results.append(parsing_result)
            meta['parsing_results'] = parsing_results

    return file_list
