import os

import cv2
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
from utils.transforms import get_affine_transform

checkpoints_path = '/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/'
ckpt_choice = {
    'lip': 'exp-schp-201908261155-lip.pth',
    'atr': 'atr_abn_checkpoint.pth',
    'pascal': 'exp-schp-201908270938-pascal-person-part.pth'
}
choices = ['lip', 'atr', 'pascal', 'torso_pascal']
ity = choices[3]
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
    },
    'torso_pascal': {
        'input_size': [512, 512],
        'num_classes': 2,
        'label': ['Background', 'Torso', ],
    },
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
                        # default='/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/pascal_abn_checkpoint.pth',
                        default='/home/qiu/Projects/Self-Correction-Human-Parsing/ckpt/schp_2_checkpoint.pth.tar',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default=inputs[inputd],
                        help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='./output_' + inputd + '_' + ity,
                        help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)


def _xywh2cs(x, y, w, h):
    aspect_ratio = 1.0
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


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

    model = networks.init_model('resnet101_1', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485],
                             std=[0.225, 0.224, 0.229])
    ])

    img_path = '/home/qiu/Projects/Self-Correction-Human-Parsing/run/garmin-forerunner-245-what-to-expect.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    # Get person center and scale
    person_center, s = _box2cs([0, 0, w - 1, h - 1])
    r = 0
    trans = get_affine_transform(person_center, s, r, input_size)
    input = cv2.warpAffine(
        img,
        trans,
        (int(input_size[1]), int(input_size[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))
    cv2.imwrite('input.png', input)
    x = transform(input)
    x = x.unsqueeze(0)

    dummy_input = torch.rand(1, 3, 512, 512).cuda()
    # x = torch.rand(1, 3, 512, 512)
    # model = model.eval()

    unscripted_output = model(x.cuda())  # Get the unscripted model's prediction...
    # unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
    # print('Python model top 5 results:\n  {}'.format(unscripted_top5))
    # script_model=torch.jit.script(model)
    traced_model = torch.jit.trace(model, dummy_input)

    scripted_output = traced_model(x.cuda())  # ...and do the same for the scripted version
    output = unscripted_output
    upsample = torch.nn.Upsample(size=input_size,
                                 mode='bilinear',
                                 align_corners=True)
    # upsample_output = upsample(output[0][-1][0].unsqueeze(0))
    upsample_output = upsample(output[0].unsqueeze(0))
    # upsample_output = output[0].unsqueeze(0)
    upsample_output = upsample_output.squeeze()
    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
    logits_result = upsample_output.data.cpu().numpy()
    parsing_result = np.argmax(logits_result, axis=2)
    cv2.imwrite('./ps.png', parsing_result * 255)
    cv2.imshow('ps', parsing_result * 255)
    cv2.waitKey(-1)
    # scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

    # print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))
    traced_model.save(ity + '_abn.pt')
    print(torch.where(torch.eq(unscripted_output, scripted_output) != 1))
    # print(torch.where(torch.eq(unscripted_output[0][1], scripted_output[0][1]) != 1))
    # print(torch.where(torch.eq(unscripted_output[1][0], scripted_output[1][0]) != 1))


if __name__ == '__main__':
    main()
