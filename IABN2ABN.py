import torch
import os

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
import torch.nn.functional as F

eps = 1e-5
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
                        default=os.path.join(checkpoints_path, ckpt_choice[ity]),
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default=inputs[inputd],
                        help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='./output_' + inputd + '_' + ity,
                        help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


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

    state_dict = torch.load(os.path.join(checkpoints_path, ckpt_choice[ity]))['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k[7:]
        ss = k.split('.')
        if ss[-2].startswith('bn') and ss[-1].endswith('weight'):
            v1 = torch.abs(v) + eps
        else:
            v1 = v
        new_state_dict[k] = v1
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    torch.save(model.state_dict(), '/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/'+ity+'_abn_checkpoint.pth')


if __name__ == '__main__':
    main()
