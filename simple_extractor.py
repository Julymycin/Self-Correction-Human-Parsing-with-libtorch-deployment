#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
from torchsummary import summary

checkpoints_path = '/home/qiu/Downloads/models/SCHP/'
ckpt_choice = {
    'lip': 'exp-schp-201908261155-lip.pth',
    'atr': 'exp-schp-201908301523-atr.pth',
    'pascal': 'exp-schp-201908270938-pascal-person-part.pth'
}
choices = ['lip', 'atr', 'pascal']
ity = choices[2]
inputs = {
    'lidingtiao': '/home/qiu/Projects/lidingtiao/full_cut/imgs/',
    'run': './run/',
    'err': './error/'
}
inputs_ch = ['lidingtiao', 'run', 'err']
inputd = inputs_ch[2]
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default=ity, choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str,
                        # default=os.path.join(checkpoints_path, ckpt_choice[ity]),
                        default='/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/pascal_abn_checkpoint.pth',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default=inputs[inputd],
                        help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='./output_' + inputd + '_' + ity,
                        help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


# def fuse_bn_to_conv(bn_layer, conv_layer):
#     # print('bn fuse')
#     bn_st_dict = bn_layer.state_dict()
#     conv_st_dict = conv_layer.state_dict()
#
#     # BatchNorm params
#     eps = bn_layer.eps
#     mu = bn_st_dict['running_mean']
#     var = bn_st_dict['running_var']
#     gamma = bn_st_dict['weight']
#
#     if 'bias' in bn_st_dict:
#         beta = bn_st_dict['bias']
#     else:
#         beta = torch.zeros(gamma.size(0)).float().to(gamma.device)
#
#     # Conv params
#     W = conv_st_dict['weight']
#     if 'bias' in conv_st_dict:
#         bias = conv_st_dict['bias']
#     else:
#         bias = torch.zeros(W.size(0)).float().to(gamma.device)
#
#     denom = torch.sqrt(var + eps)
#     b = beta - gamma.mul(mu).div(denom)
#     A = gamma.div(denom)
#     bias *= A
#     A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)
#
#     W.mul_(A)
#     bias.add_(b)
#
#     conv_layer.weight.data.copy_(W)
#     if conv_layer.bias is None:
#         conv_layer.bias = torch.nn.Parameter(bias)
#     else:
#         conv_layer.bias.data.copy_(bias)
#
#
# def extract_layers(model):
#     list_layers = []
#     for n, p in model.named_modules():
#         list_layers.append(n)
#     return list_layers
#
#
# def compute_next_bn(layer_name, resnet):
#     list_layer = extract_layers(resnet)
#     assert layer_name in list_layer
#     if layer_name == list_layer[-1]:
#         return None
#     next_bn = list_layer[list_layer.index(layer_name) + 1]
#     if extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm2d':
#         return next_bn
#     return None
#
#
# def fuse_bn(model):
#     for n, m in model.named_modules():
#         if isinstance(m, nn.Conv2d):
#             next_bn = compute_next_bn(n, model)
#             if next_bn is not None:
#                 next_bn_ = extract_layer(model, next_bn)
#                 fuse_bn_to_conv(next_bn_, m)
#                 set_layer(model, next_bn, nn.Identity())
#     return model


def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    summary(model.cuda(), (3, 512, 512))
    # state_dict = torch.load(args.model_restore)['state_dict']
    state_dict = torch.load(args.model_restore)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_palette(num_classes)
    # with torch.no_grad():
    #     inp = torch.ones([1, 3, 512, 512]).cuda()
        
    #     out = model(inp)
    #     out1 = out.squeeze()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            # upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample(output[0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)
            if args.logits:
                logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()
