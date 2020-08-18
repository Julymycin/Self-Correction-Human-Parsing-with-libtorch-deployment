#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   train.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

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
                        default='./log/checkpoint_20.pth.tar')
    parser.add_argument("--schp-start", type=int, default=21, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=5, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str,
                        default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()


def main():
    # import os
    # from PIL import Image
    # import matplotlib.pyplot as plt
    # img =Image.open('/home/qiu/Downloads/datasets/pascal_person_part/pascal_person_part_gt/2008_000026.png')
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(img)
    # plt.axis('on')  # 关掉坐标轴为 off
    # plt.title('image')  # 图像题目
    # plt.show()

    pre_ckpt = '/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/pascal_abn_checkpoint.pth'
    args = get_arguments()
    print(args)

    start_epoch = 0
    cycle_n = 0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

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
    from torchsummary import summary
    # summary(model, (3, 256, 256))

    IMAGE_MEAN = AugmentCE2P.mean
    IMAGE_STD = AugmentCE2P.std
    INPUT_SPACE = AugmentCE2P.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))

    SCHP_AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    schp_model = DataParallelModel(SCHP_AugmentCE2P)
    # schp_model=SCHP_AugmentCE2P
    # state_dict = torch.load(pre_ckpt)['state_dict']
    state_dict = torch.load(pre_ckpt)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        v1 = v
        k = 'module.' + k
        if k == 'module.decoder.conv4.weight' or k == 'module.fushion.3.weight':
            x1 = torch.zeros(torch.Size([2, 256, 1, 1])).cuda()
            for i in range(7):
                if i != 2:
                    x1[0, :] += v[i, :]
            x1 = x1 / 5
            x1[1, :] = v[2, :]
            v1 = x1
        elif k == 'module.decoder.conv4.bias' or k == 'module.fushion.3.bias':
            x2 = torch.zeros(torch.Size([2])).cuda()
            x2[0] = (torch.sum(v) - v[2]) / 6
            x2[1] = v[2]
            v1 = x2
        new_state_dict[k] = v1

    model.load_state_dict(new_state_dict)
    schp_model.load_state_dict(new_state_dict)

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

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    # Data Loader
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

    train_dataset = PascalDataSet(args.data_dir, 'train', crop_size=input_size, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size * len(gpus),
                                   num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
    print('Total training samples: {}'.format(len(train_dataset)))

    test_dataset = PascalDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform, )
    test_num_samples = len(test_dataset)
    print('Totoal testing sample numbers: {}'.format(test_num_samples))
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                 eta_min=args.learning_rate / 100, warmup_epoch=10,
                                 start_cyclical=args.schp_start, cyclical_base_lr=args.learning_rate / 2,
                                 cyclical_epoch=args.cycle_epochs)

    total_iters = args.epochs * len(train_loader)
    start = timeit.default_timer()

    for epoch in range(start_epoch, args.epochs):

        lr = lr_scheduler.get_lr()[0]

        model.train()
        for i_iter, batch in enumerate(train_loader):
            i_iter += len(train_loader) * epoch

            images, labels, _ = batch
            labels = labels.cuda(non_blocking=True)

            edges = generate_edge_tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            edges = edges.type(torch.cuda.LongTensor)

            preds = model(images)

            # Online Self Correction Cycle with Label Refinement
            if cycle_n >= 1:
                with torch.no_grad():
                    soft_preds = schp_model(images)
                    soft_parsings = soft_preds[0][-1]
                    soft_edges = soft_preds[1][-1]
                    # soft_parsing = []
                    # soft_edge = []

                    # for soft_pred in soft_preds:
                    #     soft_parsing.append(soft_pred[0][-1])
                    #     # soft_edge.append(soft_pred[1][-1])
                    #     soft_edge.append(soft_pred[1])
                    # soft_preds = torch.cat(soft_parsing, dim=0)
                    # soft_edges = torch.cat(soft_edge, dim=0)
            else:
                soft_parsings = None
                soft_edges = None

            loss = criterion(preds, [labels, edges, soft_parsings, soft_edges], cycle_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i_iter % 100 == 0:
                print('iter = {} of {} completed, lr = {}, loss = {}'.format(i_iter, total_iters, lr,
                                                                             loss.data.cpu().numpy()))
        if (epoch + 1) % (args.eval_epochs) == 0:
            schp.save_schp_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, False, args.log_dir, filename='checkpoint_{}.pth.tar'.format(epoch + 1))
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
        # Self Correction Cycle with Model Aggregation
        if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
            print('Self-correction cycle number {}'.format(cycle_n))
            schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
            cycle_n += 1
            schp.bn_re_estimate(train_loader, schp_model)
            schp.save_schp_checkpoint({
                'state_dict': schp_model.state_dict(),
                'cycle_n': cycle_n,
            }, False, args.log_dir, filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        print('epoch = {} of {} completed using {} s'.format(epoch, args.epochs,
                                                             (end - start) / (epoch - start_epoch + 1)))

    end = timeit.default_timer()
    print('Training Finished in {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
