import argparse
import cv2
import os
import json
import numpy as np
from PIL import Image as PILImage
import joblib
import copy


def mask_nms(masks,
             bbox_scores,
             instances_confidence_threshold=0.5,
             overlap_threshold=0.7):
    """
    NMS-like procedure used in Panoptic Segmentation
    Remove the overlap areas of different instances in Instance Segmentation
    """
    panoptic_seg = np.zeros(masks.shape[:2], dtype=np.uint8)
    sorted_inds = list(range(len(bbox_scores)))
    current_segment_id = 0
    segments_score = []

    for inst_id in sorted_inds:
        score = bbox_scores[inst_id]
        if score < instances_confidence_threshold:
            break
        mask = masks[:, :, inst_id]
        mask_area = mask.sum()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        #         panoptic_seg[np.where(mask==1)] = current_segment_id
        #         panoptic_seg = panoptic_seg + current_segment_id*mask
        panoptic_seg = np.where(mask == 0, panoptic_seg, current_segment_id)
        segments_score.append(score)
    #         print(np.unique(panoptic_seg))
    return panoptic_seg, segments_score


def extend(si, sj, instance_label, global_label, panoptic_seg_mask, class_map):
    """
    """
    directions = [[-1, 0], [0, 1], [1, 0], [0, -1], [1, 1], [1, -1], [-1, 1],
                  [-1, -1]]

    inst_class = instance_label[si, sj]
    human_class = panoptic_seg_mask[si, sj]
    global_class = class_map[inst_class]
    queue = [[si, sj]]

    while len(queue) != 0:
        cur = queue[0]
        queue.pop(0)

        for direction in directions:
            ni = cur[0] + direction[0]
            nj = cur[1] + direction[1]

            if ni >= 0 and nj >= 0 and \
                    ni < instance_label.shape[0] and \
                    nj < instance_label.shape[1] and \
                    instance_label[ni, nj] == 0 and \
                    global_label[ni, nj] == global_class:
                instance_label[ni, nj] = inst_class
                # Using refined instance label to refine human label
                panoptic_seg_mask[ni, nj] = human_class
                queue.append([ni, nj])


def refine(instance_label, panoptic_seg_mask, global_label, class_map):
    """
    Inputs:
        [ instance_label ]
            np.array() with shape [h, w]
        [ global_label ] with shape [h, w]
            np.array()
  """
    for i in range(instance_label.shape[0]):
        for j in range(instance_label.shape[1]):
            if instance_label[i, j] != 0:
                extend(i, j, instance_label, global_label, panoptic_seg_mask,
                       class_map)


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Inputs:
        =num_cls=
            Number of classes.
    Returns:
        The color map.
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


def patch2img_output(parsing_results, img_name, img_height, img_width, bbox,
                     bbox_type, num_class):
    """transform bbox patch outputs to image output"""
    assert bbox_type == 'gt' or 'msrcnn'
    output = np.zeros((img_height, img_width, num_class), dtype='float')
    output[:, :, 0] = np.inf
    count_predictions = np.zeros((img_height, img_width, num_class),
                                 dtype='int32')
    for i in range(len(bbox)):  # person index starts from 1
        bbox_parsing = parsing_results[i + 1]
        temp = np.zeros(bbox_parsing.shape)
        bbox_output = np.array([temp, bbox_parsing])
        bbox_output = bbox_output.transpose(1, 2, 0)

        output[bbox[i][1]:bbox[i][3] + 1, bbox[i][0]:bbox[i][2] + 1,
               1:] += bbox_output[:, :, 1:]
        count_predictions[bbox[i][1]:bbox[i][3] + 1, bbox[i][0]:bbox[i][2] + 1,
                          1:] += 1
        output[bbox[i][1]:bbox[i][3] + 1, bbox[i][0]:bbox[i][2] + 1, 0] \
            = np.minimum(output[bbox[i][1]:bbox[i][3] + 1, bbox[i][0]:bbox[i][2] + 1, 0], bbox_output[:, :, 0])

    # Caution zero dividing.
    count_predictions[count_predictions == 0] = 1
    return output / count_predictions


def get_instance(cat_gt, panoptic_seg_mask):
    """
    """
    instance_gt = np.zeros_like(cat_gt, dtype=np.uint8)
    num_humans = len(np.unique(panoptic_seg_mask)) - 1
    class_map = {}

    total_part_num = 0
    for id in range(1, num_humans + 1):
        human_part_label = np.where(panoptic_seg_mask == id, cat_gt,
                                    0).astype(np.uint8)
        #         human_part_label = (np.where(panoptic_seg_mask==id) * cat_gt).astype(np.uint8)
        part_classes = np.unique(human_part_label)

        exceed = False
        for part_id in part_classes:
            if part_id == 0:  # background
                continue
            total_part_num += 1

            if total_part_num > 255:
                print("total_part_num exceed, return current instance map: {}".
                      format(total_part_num))
                exceed = True
                break
            class_map[total_part_num] = part_id
            instance_gt[np.where(human_part_label == part_id)] = total_part_num
        if exceed:
            break

    # Make instance id continous.
    ori_cur_labels = np.unique(instance_gt)
    total_num_label = len(ori_cur_labels)
    if instance_gt.max() + 1 != total_num_label:
        for label in range(1, total_num_label):
            instance_gt[instance_gt == ori_cur_labels[label]] = label

    final_class_map = {}
    for label in range(1, total_num_label):
        if label >= 1:
            final_class_map[label] = class_map[ori_cur_labels[label]]

    return instance_gt, final_class_map


def compute_confidence(im_name, feature_map, class_map, instance_label,
                       output_dir, panoptic_seg_mask, seg_score_list):
    """
    """
    conf_file = open(
        os.path.join(output_dir,
                     os.path.splitext(im_name)[0] + '.txt'), 'w')

    weighted_map = np.zeros_like(feature_map[:, :, 0])
    for index, score in enumerate(seg_score_list):
        weighted_map += (panoptic_seg_mask == index + 1) * score

    for label in class_map.keys():
        cls = class_map[label]
        confidence = feature_map[:, :, cls].reshape(-1)[np.where(
            instance_label.reshape(-1) == label)]
        confidence = (weighted_map *
                      feature_map[:, :, cls].copy()).reshape(-1)[np.where(
                          instance_label.reshape(-1) == label)]

        confidence = confidence.sum() / len(confidence)
        conf_file.write('{} {}\n'.format(cls, confidence))

    conf_file.close()


def result_saving(fused_output, img_name, img_height, img_width, output_dir,
                  instance_masks, bbox_score, msrcnn_bbox, data_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    global_root = os.path.join(output_dir, 'global_parsing')
    instance_root = os.path.join(output_dir, 'instance_parsing')
    tag_dir = os.path.join(output_dir, 'global_tag')

    src_file = os.path.join(data_dir, 'src_imgs/' + img_name)
    src = cv2.imread(src_file)
    # src = PILImage.open(src_file)

    if not os.path.exists(global_root):
        os.makedirs(global_root)
    if not os.path.exists(instance_root):
        os.makedirs(instance_root)
    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)

    # For visualizing indexed png image.
    palette = get_palette(256)

    fused_output = cv2.resize(fused_output,
                              dsize=(img_width, img_height),
                              interpolation=cv2.INTER_LINEAR)
    seg_pred = np.asarray(np.argmax(fused_output, axis=2), dtype=np.uint8)
    # masks = np.load(mask_output_path)
    masks = copy.copy(instance_masks)
    masks[np.where(seg_pred == 0)] = 0

    instance_test = seg_pred*instance_masks
    instance_test = cv2.cvtColor(instance_test, cv2.COLOR_GRAY2BGR)
    instance_test[:, :, 0] = np.where(
        instance_test[:, :, 0] != 0,
        255 / instance_test.max() * instance_test[:, :, 0], 0)
    instance_test[:, :, 1] = np.where(
        instance_test[:, :, 1] != 0,
        255 / instance_test.max() * instance_test[:, :, 1], 0)
    cv2.imwrite(
        os.path.join(instance_root,
                     os.path.splitext(img_name)[0] + '_test.png'), instance_test)

#     panoptic_seg_mask = masks
#     seg_score_list = bbox_score

#     instance_pred, class_map = get_instance(seg_pred, panoptic_seg_mask)
#     # refine(instance_pred, panoptic_seg_mask, seg_pred, class_map)

#     compute_confidence(img_name, fused_output, class_map, instance_pred,
#                        instance_root, panoptic_seg_mask, seg_score_list)

#     # ins_seg_results = open(
#     #     os.path.join(tag_dir,
#     #                  os.path.splitext(img_name)[0] + '.txt'), "a")
#     # keep_human_id_list = list(np.unique(panoptic_seg_mask))
#     # if 0 in keep_human_id_list:
#     #     keep_human_id_list.remove(0)
#     # for i in keep_human_id_list:
#     #     ins_seg_results.write('{:.6f} {} {} {} {}\n'.format(
#     #         seg_score_list[i - 1], int(msrcnn_bbox[i - 1][1]),
#     #         int(msrcnn_bbox[i - 1][0]), int(msrcnn_bbox[i - 1][3]),
#     #         int(msrcnn_bbox[i - 1][2])))
#     # ins_seg_results.close()

#     output_im_global = PILImage.fromarray(seg_pred)
#     output_im_instance = PILImage.fromarray(instance_pred)
#     output_im_tag = PILImage.fromarray(panoptic_seg_mask)
#     output_im_global.putpalette(palette)
#     output_im_instance.putpalette(palette)
#     output_im_tag.putpalette(palette)

#     # tt = cv2.cvtColor(instance_pred, cv2.COLOR_GRAY2BGR)

#     src[:, :, 0] = np.where(instance_pred == 0, src[:, :, 0], 255)
#     src[:, :, 1] = np.where(masks == 0, src[:, :, 1], 255)
#     src[:, :, 2] = np.where(instance_masks == 0, src[:, :, 2], 255)
#     # src[np.where(instance_pred != 0), 0] = 255

#     output_im_global.save(
#         os.path.join(global_root,
#                      os.path.splitext(img_name)[0] + '.png'))
#     output_im_instance.save(
#         os.path.join(instance_root,
#                      os.path.splitext(img_name)[0] + '.png'))
#     cv2.imwrite(
#         os.path.join(instance_root,
#                      os.path.splitext(img_name)[0] + '_mask.png'), src)
#     output_im_tag.save(
#         os.path.join(tag_dir,
#                      os.path.splitext(img_name)[0] + '.png'))


def multi_process(a, mask_dir, save_dir, data_dir):
    img_name = a['im_name']
    img_height = a['img_height']
    img_width = a['img_width']
    msrcnn_bbox = a['person_bbox']
    bbox_score = a['person_bbox_score']
    parsing_results = a['parsing_results']
    instance_masks = a['instance_masks']

    global_parsing = parsing_results[0]
    temp = np.zeros(global_parsing.shape)
    global_output = np.array([temp, global_parsing])
    global_output = global_output.transpose(1, 2, 0)

    # msrcnn_output = patch2img_output(args.mask_output_dir, img_name, img_height, img_width, msrcnn_bbox,
    #                                  bbox_type='msrcnn', num_class=20)

    local_output = patch2img_output(parsing_results,
                                    img_name,
                                    img_height,
                                    img_width,
                                    msrcnn_bbox,
                                    bbox_type='msrcnn',
                                    num_class=2)

    #### global and local branch logits fusion #####
    #     fused_output = global_output + msrcnn_output + gt_output
    fused_output = global_output + local_output

    result_saving(fused_output, img_name, img_height, img_width, save_dir,
                  instance_masks, bbox_score, msrcnn_bbox, data_dir)
    return


# def fuse_process():


def gl_fuse(
    parsing_list,
    mask_dir,
    save_dir,
    data_dir,
):

    # results = joblib.Parallel(n_jobs=8, verbose=0, pre_dispatch=16)([
    #     joblib.delayed(multi_process)(a, mask_dir, save_dir)
    #     for i, a in enumerate(parsing_list)
    # ])
    for i, a in enumerate(parsing_list):
        multi_process(a, mask_dir, save_dir, data_dir)