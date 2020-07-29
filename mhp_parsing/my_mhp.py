from coco_style_annotation_creator import test_human2coco_format as h2c
from detect2.tools import myinfer
import make_crop_and_mask_w_mask_nms as mcm
from global_local_parsing import global_local_evaluate as gleval
import logits_fusion as lfs
import argparse
import os
import time




def h2c_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='test', help="name of dataset for register")
    parser.add_argument("--root_dir", type=str, default='./mhp_parsing/data/')

    # for instance detection
    parser.add_argument("--config_file", type=str, default='./mhp_parsing/detect2/configs/Misc/demo.yaml', help='detectron2 config file')
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")


    return parser.parse_args()




def main():
    count=[]
    time0=time.time()

    args1=h2c_arguments()
    dataname=args1.dataset
    # no-label test imgs -> coco Dataset
    data_dir=os.path.join(args1.root_dir,dataname)
    src_dir=os.path.join(data_dir,'src_imgs')
    anno_dir=os.path.join(data_dir,'src_annos')
    if not os.path.exists(anno_dir):
        os.mkdir(anno_dir)
    h2c.tococo(src_dir, anno_dir, dataname)

    time1=time.time()
    count.append(time1-time0)

    # get the instance result
    instance_det_out=os.path.join(data_dir,'instance_detection')
    anno_file=os.path.join(anno_dir,dataname+'.json')
    if not os.path.exists(instance_det_out):
        os.mkdir(instance_det_out)
    cfg_modify=['MODEL.WEIGHTS', '/home/qiu/Downloads/models/detectron2/detectron2_maskrcnn_cihp_finetune.pth', 
                'TEST.AUG.ENABLED', 'False', 
                'DATALOADER.NUM_WORKERS', '8', 
                'DATASETS.TRAIN', (dataname+'_train',),
                'DATASETS.TEST', (dataname+'_val',),
                'OUTPUT_DIR',instance_det_out,]
    myinfer.infer(args1,cfg_modify, src_dir,anno_file)

    time2=time.time()
    count.append(time2-time1)

    #crop the instances
    mcm.crop(src_dir,data_dir,anno_file,instance_det_out)

    time3=time.time()
    count.append(time3-time2)

    #use the same pretrained checkpoint for local and global parsing
    schp_ckpt='/home/qiu/Projects/Self-Correction-Human-Parsing/deploy/pascal_abn_checkpoint.pth'
    gleval.glparsing(data_dir,['crop_pic','src_imgs'],schp_ckpt,log_dir=data_dir,)

    time4=time.time()
    count.append(time4-time3)

    #fuse result
    crop_json=os.path.join(data_dir,'crop.json')
    gp_dir=os.path.join(data_dir,'src_imgs_parsing')
    lp_dir=os.path.join(data_dir,'crop_pic_parsing')
    mask_dir=os.path.join(data_dir,'crop_mask')
    save_dir=os.path.join(data_dir,'fusion_result')
    lfs.gl_fuse(crop_json,gp_dir,lp_dir,mask_dir,save_dir)

    time5=time.time()
    count.append(time5-time4)
    count.append(time5-time0)

    print(count)






if __name__=='__main__':
    # start=time.time()
    main()
    # print(time.time()-start)



    
