import os
import json
import timeit
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils import data
from tqdm import tqdm

import networks
import utils.schp as schp
from datasets.datasets import LIPDataSet, PascalDataSet
from datasets.target_generation import generate_edge_tensor
from utils.transforms import BGR2RGB_transform
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler
import numpy as np
from evaluate import multi_scale_testing
from utils.miou import compute_mean_ioU


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101_3')
    # Data Preference
    parser.add_argument("--data-dir", type=str,
                        # default='/home/qiu/Downloads/datasets/LIP',
                        # default='/home/qiu/Downloads/datasets/ICCV15_fashion_dataset(ATR)/humanparsing',
                        default='/home/qiu/Downloads/datasets/pascal_person_part',
                        )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=str,
                        # default='473,473'
                        default='512,512'
                        # default='256,256'
                        )
    parser.add_argument("--num-classes", type=int, default=2)
    # parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-epochs", type=int, default=5)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str,
                        default='./log/checkpoint_30.pth.tar')
    parser.add_argument("--schp-start", type=int, default=21, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=5, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str,
                        default='./log/schp_2_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()


def main():
    args = get_arguments()
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = list(map(int, args.input_size.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model Initialization
    AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    model = DataParallelModel(AugmentCE2P)
    # model=AugmentCE2P
    model.cuda()

    SCHP_AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    schp_model = DataParallelModel(SCHP_AugmentCE2P)
    schp_model.cuda()

    restore_from = args.model_restore
    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))
        checkpoint = torch.load(restore_from)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        # start_epoch = 0
    if os.path.exists(args.schp_restore):
        print('Resuming schp checkpoint from {}'.format(args.schp_restore))
        schp_checkpoint = torch.load(args.schp_restore)
        schp_model_state_dict = schp_checkpoint['state_dict']
        cycle_n = schp_checkpoint['cycle_n']
        # cycle_n = 0
        schp_model.load_state_dict(schp_model_state_dict)

    # Data Loader
    IMAGE_MEAN = AugmentCE2P.mean
    IMAGE_STD = AugmentCE2P.std
    INPUT_SPACE = AugmentCE2P.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    elif INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    test_dataset = PascalDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform, )
    test_num_samples = len(test_dataset)
    print('Totoal testing sample numbers: {}'.format(test_num_samples))
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    parsing_preds = []
    gts = []
    scales = np.zeros((test_num_samples, 2), dtype=np.float32)
    centers = np.zeros((test_num_samples, 2), dtype=np.int32)
    interp = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(tqdm(testloader)):
            image, gt, meta = batch
            if (len(image.shape) > 4):
                image = image.squeeze()
            im_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            scales[idx, :] = s
            centers[idx, :] = c
            parsing_output = model(image.cuda())
            parsing_output = parsing_output[0][-1]
            output = parsing_output[0]
            ms_fused_parsing_output = interp(output.unsqueeze(0))
            ms_fused_parsing_output = ms_fused_parsing_output[0].permute(1, 2, 0)  # HWC
            parsing = torch.argmax(ms_fused_parsing_output, dim=2)
            parsing = parsing.data.cpu().numpy()
            parsing_preds.append(parsing)
            gts.append(gt)
    assert len(parsing_preds) == test_num_samples
    mIoU = compute_mean_ioU(parsing_preds, gts, scales, centers, args.num_classes, args.data_dir, input_size)
    print(mIoU)

    parsing_preds = []
    gts = []
    scales = np.zeros((test_num_samples, 2), dtype=np.float32)
    centers = np.zeros((test_num_samples, 2), dtype=np.int32)
    interp = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    with torch.no_grad():
        schp_model.eval()
        for idx, batch in enumerate(tqdm(testloader)):
            image, gt, meta = batch
            if (len(image.shape) > 4):
                image = image.squeeze()
            im_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            scales[idx, :] = s
            centers[idx, :] = c
            parsing_output = schp_model(image.cuda())
            parsing_output = parsing_output[0][-1]
            output = parsing_output[0]
            ms_fused_parsing_output = interp(output.unsqueeze(0))
            ms_fused_parsing_output = ms_fused_parsing_output[0].permute(1, 2, 0)  # HWC
            parsing = torch.argmax(ms_fused_parsing_output, dim=2)
            parsing = parsing.data.cpu().numpy()
            parsing_preds.append(parsing)
            gts.append(gt)
    assert len(parsing_preds) == test_num_samples
    mIoU = compute_mean_ioU(parsing_preds, gts, scales, centers, args.num_classes, args.data_dir, input_size)
    print(mIoU)


if __name__ == '__main__':
    main()
