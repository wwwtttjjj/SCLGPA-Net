import numpy as np
from PIL import Image, ImageDraw
import json
import cv2
import os
import math
import albumentations

labels2color = {"PED": 100, "SRF": 150, "IRF": 255}
# trans = albumentations.Compose([albumentations.CLAHE(clip_limit=4.0,tile_grid_size=(4, 4), p=0.9)])

'''median'''
def median_Blur_gray(img, filiter_size = 3):  #当输入的图像为灰度图像
    image_copy = np.array(img, copy = True).astype(np.float32)
    processed = np.zeros_like(image_copy)
    middle = int(filiter_size / 2)
    
    for i in range(middle, image_copy.shape[0] - middle):
        for j in range(middle, image_copy.shape[1] - middle):
            temp = []
            for m in range(i - middle, i + middle +1):
                for n in range(j - middle, j + middle + 1):
                    if m-middle < 0 or m+middle+1 >image_copy.shape[0] or n-middle < 0 or n+middle+1 > image_copy.shape[1]:
                        temp.append(0)
                    else:
                        temp.append(image_copy[m][n])
                    #count += 1
            temp.sort()
            processed[i][j] = temp[(int(filiter_size*filiter_size/2)+1)]
    processed = processed.astype(np.uint64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0 or j == 0 or i == img.shape[0] - 1 or j == img.shape[1] - 1:
                processed[i][j] = img[i][j]
    return processed
'''usm sharping'''


def usm_edge_sharpening(img):
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    usm = cv2.addWeighted(img, 2.0, blur_img, -0.5, 0)
    return usm


'''two point, get PED pixel index'''


def get_PED_index(start, end):
    points = []
    start_x = start[0]
    start_y = start[1]
    end_x = end[0]
    end_y = end[1]
    delta_x = end_x - start_x
    delta_y = end_y - start_y

    if abs(delta_x) > abs(delta_y):
        steps = abs(delta_x)
    else:
        steps = abs(delta_y)

    x_step = delta_x / steps
    y_step = delta_y / steps

    x = start_x
    y = start_y
    while steps >= 0:
        points.append([round(x), round(y)])
        x += x_step
        y += y_step
        steps -= 1
    return points


'''two type of data include points and line'''


def create_index(json_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    W, H = data['imageWidth'], data['imageHeight']
    # mask = Image.fromarray(np.zeros((H, W))).convert('L')
    mask_blank = np.array(Image.fromarray(np.zeros((H, W))).convert('L'))
    # mask1 = ImageDraw.Draw(mask)
    PED_index = []
    SRF_IRF_index = {}
    for points in data['shapes']:
        if points['label'] == 'PED':
            for [x, y] in points["points"]:
                PED_index.append([int(y), int(x)])
        else:
            [[x, y]] = points["points"]
            SRF_IRF_index[(int(y), int(x))] = labels2color[points['label']]
        # mask1.polygon(xy, fill = (labels2color[points['label']]))
    return PED_index, SRF_IRF_index, mask_blank


'''get the masked of PED'''


def get_PED_mask(PED_index, label_slic):
    PED_mask = {}
    PED_short_mask = {}
    slope_mask = {}
    slope_short_mask = {}

    for i in range(0, len(PED_index), 2):
        PED_xy = []
        start = PED_index[i]
        end = PED_index[i + 1]
        if start[1] > end[1]:
            start, end = end, start
        A = end[1] - start[1]#y2-y1
        B = start[0] - end[0]#x1-x2
        C = end[0] * start[1] - start[0] * end[1]#x2 * y1 - x1 * y2
        PED_xy += get_PED_index(PED_index[i], PED_index[i + 1])
        if len(PED_xy) < 3:
            for [x, y] in PED_xy:
                PED_short_mask[label_slic[x][y]] = labels2color['PED']
                slope_short_mask[label_slic[x][y]] = [A, B, C]
        else:
            for [x, y] in PED_xy:
                PED_mask[label_slic[x][y]] = labels2color['PED']
                slope_mask[label_slic[x][y]] = [A, B, C]
        
    return PED_mask, PED_short_mask, {**slope_mask, **slope_short_mask}


'''get the masked of SRF and IRF'''


def get_SRF_IRF_mask(SRF_IRF_index, label_slic):
    SRF_IRF_mask = {}
    for key, value in SRF_IRF_index.items():
        SRF_IRF_mask[label_slic[key[0]][key[1]]] = value
    return SRF_IRF_mask
def sin(point_1, point_2):
    return (point_1[0] - point_2[0]) / math.sqrt((point_2[1] - point_1[1]) **2 + (point_2[0] - point_1[0]) ** 2 + 0.01)

'''generate the slic superpixel img'''


def create_SLIC_image(img_path, json_path, region_size=20, ruler=20, iterate=10):
    img = cv2.imread(img_path)
    PED_index, SRF_IRF_index, mask_blank = create_index(json_path)
    if not SRF_IRF_index:
        img = usm_edge_sharpening(img)
    # sample = trans(image = img)#CACLE
    # img = sample['image']
    #初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
    slic = cv2.ximgproc.createSuperpixelSLIC(img,
                                             region_size=region_size,
                                             ruler=ruler)
    slic.iterate(iterate)  #迭代次数，越大效果越好
    mask_slic = slic.getLabelContourMask()  #获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()  #获取超像素标签

    label_slic = median_Blur_gray(label_slic, filiter_size=3)
    number_slic = slic.getNumberOfSuperpixels()  #获取超像素数目
    # print(number_slic)
    # print(len(np.unique(label_slic)))
    # mask_inv_slic = cv2.bitwise_not(mask_slic)
    # img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) #在原图上绘制超像素边界
    # cv2.imwrite('1.jpg',img_slic)
    W, H = label_slic.shape[:2]
    clsuters = [[] for _ in range(number_slic)]  # save the cluster
    neigbor_up = [[] for _ in range(number_slic)]  #save the neigbor relation
    xy_center = []  #每个聚类的中心点

    for x in range(W):
        for y in range(H):
            clsuters[label_slic[x][y]].append([x, y])

    for i in range(number_slic):
        if clsuters[i] != []:
            x_center = int(np.median([x_[0] for x_ in clsuters[i]]))
            y_center = int(np.median([y_[1] for y_ in clsuters[i]]))
            xy_center.append([x_center, y_center])
        else:
            xy_center.append([])

    neigbor_all = [[] for _ in range(number_slic)]  #save the neigbor relation
    for x in range(W):
        for y in range(H):
            if mask_slic[x][y] == 255:
                for dx, dy in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0),
                               (1, 1), (0, 1), (-1, 1)]:
                    n_x, n_y = x + dx, y + dy
                    if n_x >= 0 and n_x < W and n_y >= 0 and n_y < H:
                        if label_slic[n_x][n_y] != label_slic[x][
                                y] and label_slic[n_x][n_y] not in neigbor_all[
                                    label_slic[x][y]]:
                            # print(1)
                            # break
                            neigbor_all[label_slic[x][y]].append(
                                label_slic[n_x][n_y])
                        if label_slic[n_x][n_y] != label_slic[x][
                                y] and label_slic[n_x][n_y] not in neigbor_up[
                                    label_slic[x][y]] and sin(xy_center[label_slic[x][y]], xy_center[label_slic[n_x][n_y]]) >= 0.5:
                            # print(1)
                            # break
                            neigbor_up[label_slic[x][y]].append(
                                label_slic[n_x][n_y])

    #get the center of each superpixel

    return clsuters, neigbor_up, neigbor_all, label_slic, img, xy_center, PED_index, SRF_IRF_index, mask_blank


'''compute the dist matchscore (crop,mask)'''


# def get_crop_mask(key, clsuters):
#     minx, miny, maxx, maxy = float('inf'), float('inf'), 0, 0
#     for [x, y] in clsuters[key]:
#         if x < minx:
#             minx = x
#         if x > maxx:
#             maxx = x
#         if y < miny:
#             miny = y
#         if y > maxy:
#             maxy = y
#     mask = np.zeros((maxx - minx + 1, maxy - miny + 1), dtype=np.uint8)
#     for [x, y] in clsuters[key]:
#         mask[x - minx][y - miny] = 1
#     return minx, miny, maxx, maxy, mask


'''compute the match_score (dist)'''


# def get_hist_dice(key, current_key, clsuters, img):
#     minx_c, miny_c, maxx_c, maxy_c, mask_c = get_crop_mask(
#         current_key, clsuters)
#     minx_k, miny_k, maxx_k, maxy_k, mask_k = get_crop_mask(key, clsuters)

#     img_c = img[minx_c:maxx_c + 1, miny_c:maxy_c + 1]
#     img_k = img[minx_k:maxx_k + 1, miny_k:maxy_k + 1]
#     # img_c = img.crop((miny_c, minx_c, maxy_c + 1, maxx_c + 1))
#     # img_k = img.crop((miny_k, minx_k, maxy_k + 1, maxx_k + 1))

#     # img_c =  cv2.cvtColor(np.asarray(img_c),cv2.COLOR_RGB2BGR)
#     # img_k = cv2.cvtColor(np.asarray(img_k),cv2.COLOR_RGB2BGR)

#     img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
#     img_k = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)

#     H_c = cv2.calcHist([img_c], [0], mask_c, [256], [0, 256])
#     H_c = cv2.normalize(H_c, H_c, 0, 1, cv2.NORM_MINMAX, -1)

#     H_k = cv2.calcHist([img_k], [0], mask_k, [256], [0, 256])
#     H_k = cv2.normalize(H_k, H_k, 0, 1, cv2.NORM_MINMAX, -1)
#     match_score = cv2.compareHist(H_c, H_k, method=cv2.HISTCMP_CORREL)
#     return match_score


'''cos dice'''


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def get_cosin_dice(key, current_key, clsuters, img):
    img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frequent_key = np.zeros(256)
    frequent_cur_key = np.zeros(256)
    for [x, y] in clsuters[key]:
        frequent_key[img_gary[x][y]] += 1
    for [x, y] in clsuters[current_key]:
        frequent_cur_key[img_gary[x][y]] += 1

    return cosine_similarity(frequent_cur_key, frequent_key)

'''compute the distance of point and point'''


def p_p_distance(center_a, center_b):
    return math.sqrt((center_a[0] - center_b[0])**2 +
                     (center_a[1] - center_b[1])**2)

def set_probality(cluster, probability_map, distance, current_key):
    probability = max(1.0 - 0.1 * ((distance) // 2), 0.0)
    for [x, y] in cluster[current_key]:
        probability_map[x][y] = probability
    return probability_map
'''Expand the field of pixel blocks corresponding to weak labels(PED)'''


def get_PED_labels(masked_index,
                   neigbor,
                   clsuters,
                   img,
                   xy_center,
                   probability_map,
                   proximity_distance,
                   Threshold=0.5):
    if len(masked_index) < 3:
        return masked_index, probability_map
    stack = []
    truth_mask = {}
    # add_t = 0.001
    for key, value in masked_index.items():
        # already.append(key)
        stack.append(key)
        truth_mask[key] = value
        num = 0
        while stack:
            current_key = stack.pop(0)
            dice = get_cosin_dice(key, current_key, clsuters, img)
            if dice >= Threshold:
                neigbor_keys = neigbor[current_key]
                for neigbor_key in neigbor_keys:
                    if neigbor_key not in masked_index:
                    # if neigbor_key not in already:
                    #     already.append(neigbor_key)
                        stack.append(neigbor_key)
                truth_mask[current_key] = value
                num += 1
                if key != current_key:
                    distance = p_p_distance(xy_center[key], xy_center[current_key]) // proximity_distance + 1#几格距离
                    probability_map = set_probality(clsuters, probability_map, distance, current_key)
            if num >= 70:
                stack = []
                break
            else:
                continue
    return truth_mask, probability_map


'''Expand the field of pixel blocks corresponding to weak labels(SRF and IRF)'''


def get_SRF_IRF_labels(masked_index,
                       truth_PED_mask,
                       neigbor,
                       clsuters,
                       img,
                       xy_center,
                       probability_map,
                       proximity_distance,
                       Threshold=0.5):
    stack = []
    already = []
    truth_mask = {}
    # add_t = 0.001
    for key, value in masked_index.items():
        already.append(key)
        stack.append(key)
        truth_mask[key] = value
        threshold = Threshold
        while stack:
            current_key = stack.pop(0)
            dice = get_cosin_dice(key, current_key, clsuters, img)
            # print(key, current_key, dice)
            if dice >= threshold:
                neigbor_keys = neigbor[current_key]
                # print(neigbor_keys)
                for neigbor_key in neigbor_keys:
                    if neigbor_key not in already and neigbor_key not in truth_PED_mask:
                        already.append(neigbor_key)
                        stack.append(neigbor_key)
                truth_mask[current_key] = value

                if key != current_key:
                    distance = p_p_distance(xy_center[key], xy_center[current_key]) // proximity_distance + 1#几格距离
                    probability_map = set_probality(clsuters, probability_map, distance, current_key)
            else:
                continue
    return truth_mask, probability_map


'''fill the holes'''


def fill_holes(clsuters, neigbor, truth_mask, probability_map):
    probabilty = []
    add_mask = {}

    # truth_mask_fill_hole = truth_mask
    for key, value in truth_mask.items():
        for n in neigbor[key]:
            # if n not in truth_mask:
            probabilty.append(n)
    for p in probabilty:
        l = len(neigbor[p])
        num = 0
        for n in neigbor[p]:
            if n in truth_mask and num == 0:
                value = truth_mask[n]
                num += 1
            elif n in truth_mask and num != 0:
                if truth_mask[n] == value:
                    num += 1
                else:
                    break
            else:
                break
        if num == l - 1:
            add_mask[p] = value
            probability_map = set_probality(clsuters, probability_map, 1, p)#fill的块修改为1
    return {**truth_mask, **add_mask}, probability_map


# mend some pixels
def mend(mask, probability_map):
    mask_mend = cv2.medianBlur(mask, 5)
    probability_map = np.uint8(probability_map * 10)
    probability_map = cv2.medianBlur(probability_map, 5)
    # print(probability_map[2-1:2+1,2-1:2+1])
    # index = np.array(mask != mask_mend)
    # for x in range(1, index.shape[0] - 1):
    #     for y in range(1, index.shape[1] - 1):
    #         if index[x][y]:
    #             probability_map = np.median(probability_map[x-1:x+1,y-1:y+1])

    return mask_mend, probability_map


#delete some nei PED
def get_detach_PED(truth_PED_mask, neigbor, truth_SRF_IRF_mask, clsuters, probability_map):
    del_key = []

    # truth_mask_fill_hole = truth_mask
    for key, value in truth_PED_mask.items():
        for n in neigbor[key]:
            if n in truth_SRF_IRF_mask:
                del_key.append(key)
                continue
    for k in del_key:
        truth_PED_mask.pop(k, None)
        probability_map = set_probality(clsuters, probability_map, 10, k)
    return truth_PED_mask
