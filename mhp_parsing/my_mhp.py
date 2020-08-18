from coco_style_annotation_creator import test_human2coco_format as h2c
from detect2.tools import myinfer
import make_crop_and_mask_w_mask_nms as mcm
from global_local_parsing import gl_eval as gleval
import parsing_fusion as pfs
import argparse
import os
import time
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import glob
import tqdm
import cv2
from detectron2.data.detection_utils import read_image


def h2c_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default='test',
                        help="name of dataset for register")
    parser.add_argument("--root_dir", type=str, default='./mhp_parsing/data/')

    # for instance detection
    parser.add_argument(
        "--config_file",
        type=str,
        # default='./mhp_parsing/detect2/configs/Misc/demo.yaml',
        default=
        './mhp_parsing/detect2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_copy.yaml',
        help='detectron2 config file')
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="perform evaluation only")

    return parser.parse_args()


def setup(args, cfg_modify=[]):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(cfg_modify)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.freeze()
    return cfg


def main():
    count = []
    time0 = time.time()

    args1 = h2c_arguments()
    dataname = args1.dataset
    # no-label test imgs -> coco Dataset
    data_dir = os.path.join(args1.root_dir, dataname)
    src_dir = os.path.join(data_dir, 'src_imgs')
    # anno_dir = os.path.join(data_dir, 'src_annos')
    # if not os.path.exists(anno_dir):
    # os.mkdir(anno_dir)
    # h2c.tococo(src_dir, anno_dir, dataname)

    time1 = time.time()
    count.append(time1 - time0)

    # get the instance result

    # instance_det_out = os.path.join(data_dir, 'instance_detection')
    # anno_file = os.path.join(anno_dir, dataname + '.json')
    # if not os.path.exists(instance_det_out):
    # os.mkdir(instance_det_out)
    # cfg_modify = [
    #     'MODEL.WEIGHTS',
    #     # '/home/qiu/Downloads/models/detectron2/detectron2_maskrcnn_cihp_finetune.pth',
    #     '/home/qiu/Downloads/models/detectron2/detectron2_mask_rcnn_fpn_3x_pre.pkl',
    # ]

    # myinfer.infer(args1, cfg_modify, src_dir, anno_file)
    cfg = setup(args1, )
    demo = DefaultPredictor(cfg)
    imgnames = os.listdir(src_dir)
    inputs = []
    for name in imgnames:
        if name.endswith(('.jpg', '.png')):
            inputs.append(os.path.join(src_dir, name))
    if len(inputs) == 1:
        inputs = glob.glob(os.path.expanduser(inputs[0]))
        assert inputs, "The input path(s) was not found"
    detect_list = []
    for path in tqdm.tqdm(inputs):
        # use PIL, to be consistent with evaluation
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # img=read_image(path, 'BGR')
        h, w, _ = img.shape
        predictions = demo(img)
        predictions['img_path'] = path
        detect_list.append(predictions)

    time2 = time.time()
    count.append(time2 - time1)

    # crop the instances
    # file_list = mcm.crop(src_dir, data_dir, anno_file, instance_det_out)
    file_list = mcm.crop(src_dir, data_dir, detect_list)

    time3 = time.time()
    count.append(time3 - time2)

    # use the same pretrained checkpoint for local and global parsing
    schp_ckpt = '/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/pascal_abn_checkpoint.pth'
    file_list = gleval.glparsing(data_dir, ['crop_pic', 'src_imgs'],
                                 schp_ckpt,
                                 log_dir=data_dir,
                                 file_list=file_list)

    time4 = time.time()
    count.append(time4 - time3)

    # fuse result

    mask_dir = os.path.join(data_dir, 'crop_mask')
    save_dir = os.path.join(data_dir, 'fusion_result')
    pfs.gl_fuse(
        file_list,
        mask_dir,
        save_dir,
        data_dir,
    )

    time5 = time.time()
    count.append(time5 - time4)
    count.append(time5 - time0)

    print(count)


if __name__ == '__main__':
    # start=time.time()
    main()
    # print(time.time()-start)
