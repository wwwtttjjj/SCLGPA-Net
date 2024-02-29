import numpy as np
from PIL import Image
import json
import cv2
import os
import glob
import argparse
import utils
from tqdm import tqdm
import math


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PED_threshold',
                        type=float,
                        default=0.45,
                        help='the threshold of hist compare distance [0, 1]')
    parser.add_argument('--SRF_IRF_threshold',
                        type=float,
                        default=0.65,
                        help='the threshold of hist compare distance [0, 1]')

    parser.add_argument('--region_size',
                        type=int,
                        default=13,
                        help='the region_size of superpixel')
    parser.add_argument('--ruler',
                        type=int,
                        default=14,
                        help='the ruler of superpixel')
    parser.add_argument('--data_path',
                        type=str,
                        default='0/test_data',
                        help='the datasets of images and jsons')
    parser.add_argument('--save_tar',
                        type=int,
                        default=0,
                        help='the datasets of images and jsons')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    save_tar = args.save_tar
    '''path'''
    jpg_paths = []
    json_paths = []

    for file in glob.glob(os.path.join(args.data_path, '*.png')):

        jpg_paths.append(file)
        json_paths.append(file[:-3] + 'json')

    for i in tqdm(range(len(jpg_paths))):

        image_path = jpg_paths[i]
        json_path = json_paths[i]
        # if os.path.exists('data/pseudo_label/' +
        #                            image_path.split('\\')[-1][:-4] + '.png'):
        #     continue
        

        img = cv2.imread(image_path)
        W, H = img.shape[0], img.shape[1]
        probability_map = np.array(Image.fromarray(np.ones(
            (W, H))).convert('L'),dtype=float)
        probability_map = probability_map / 2#初始值设置为0.5

        #全阴性
        if not os.path.isfile(json_path):
            mask_blank = np.array(
                Image.fromarray(np.zeros(
                    (W, H))).convert('L'))  #无标注返回全0的mask_blank
            Image.fromarray(mask_blank).save(str(save_tar) + '/pseudo_label/' +
                                             image_path.split('\\')[-1][:-4] +
                                             '.png')
            np.save(
                str(save_tar)+'/probability_map/' + image_path.split('\\')[-1][:-4] +
                '.npy', np.uint8(probability_map * 10))

            continue

        proximity_distance = (1.25) * args.region_size * math.sqrt(2)  #标准一格距离度量

        clsuters, neigbor_up, neigbor_all, label_slic, img, xy_center, PED_index, SRF_IRF_index, mask_blank = utils.create_SLIC_image(
            image_path, json_path, args.region_size, args.ruler)
        PED_mask, PED_short_mask, slope = utils.get_PED_mask(PED_index, label_slic)
        line_PED = {**PED_mask, **PED_short_mask}
        SRF_IRF_mask = utils.get_SRF_IRF_mask(SRF_IRF_index, label_slic)

        truth_PED_mask, probability_map = utils.get_PED_labels(
            PED_mask,
            neigbor_up,
            clsuters,
            img,
            xy_center,
            probability_map,
            proximity_distance,
            Threshold=args.PED_threshold,
        )
        truth_SRF_IRF_mask, probability_map = utils.get_SRF_IRF_labels(
            SRF_IRF_mask,
            truth_PED_mask,
            neigbor_all,
            clsuters,
            img,
            xy_center,
            probability_map,
            proximity_distance,
            Threshold=args.SRF_IRF_threshold)
        #把标注的那些label设置为1
        for k,value in {**line_PED, **SRF_IRF_mask}.items():
            utils.set_probality(clsuters, probability_map, 0, k)

        truth_PED_mask = utils.get_detach_PED(truth_PED_mask, neigbor_all,
                                              truth_SRF_IRF_mask, clsuters, probability_map)
        truth_mask = {**PED_short_mask, **truth_PED_mask, **truth_SRF_IRF_mask}
        truth_mask, probability_map = utils.fill_holes(clsuters, neigbor_all, truth_mask, probability_map)
        for key, value in truth_mask.items():
            for positions in clsuters[key]:
                mask_blank[positions[0]][positions[1]] = value
        # 把PED线段下方的points消掉
        for key, value in line_PED.items():
                for positions in clsuters[key]:
                    A = slope[key][0]
                    B = slope[key][1]
                    C = slope[key][2]
                    slope_cur = A * positions[0] + B * positions[1] + C
                    if slope_cur >= 0 :
                        mask_blank[positions[0]][positions[1]] = 0
                        probability_map[positions[0]][positions[1]] = 0.5

        # Image.fromarray(mask_blank).show()
        mask, probability_map = utils.mend(mask_blank, probability_map)


        Image.fromarray(mask).save(str(save_tar)+'/pseudo_label/' +
                                   image_path.split('\\')[-1][:-4] + '.png')

        np.save(
            str(save_tar)+'/probability_map/' + image_path.split('\\')[-1][:-4] + '.npy',
            probability_map)