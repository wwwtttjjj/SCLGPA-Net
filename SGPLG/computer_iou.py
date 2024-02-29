#mean iou 计算
import numpy as np
import os
from skimage import io
import argparse

'''
该脚本主要实现语义分割中多类结果的评估功能
要求：预测结果文件夹和真值文件夹中各个图像的文件名应该一样，对同一种类像素的灰度表示也应该一样
'''
def get_IOU(W, H, pre_path, gt_path):
    img_size = (W, H)                          #图像的尺寸（只需要长宽）
    classes = np.array([0, 100, 200, 255]).astype('uint8')#每一类的灰度值表示
    files = os.listdir(pre_path)

    res = []
    for clas in classes:

        D = np.zeros([len(files), img_size[0], img_size[1], 2]).astype(bool)#存储每一类的二值数据
        # print(D.shape)
        for i, file in enumerate(files):
            img1 = io.imread(os.path.join(pre_path, file), as_gray=True)#以灰度值的形式读取
            img2 = io.imread(os.path.join(gt_path, file), as_gray=True)#以灰度值的形式读取
            D[i, :, :, 0] = img1 == clas
            D[i, :, :, 1] = img2 == clas
        res.append(np.sum(D[..., 0] & D[..., 1])/np.sum(D[..., 0] | D[..., 1])) #计算IOU
        # print(res)
    #结果输出
    for i, clas in enumerate(classes):
        print("Class "+str(clas)+' :'+str(res[i]))
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--W', type = int, default=250)
    parser.add_argument('--H', type = int, default=600)
    parser.add_argument('--pre_path', type = str, default='data/pseudo_label')
    parser.add_argument('--gt_path', type = str, default='data/label')

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()
    get_IOU(W = args.W, H = args.H, pre_path = args.pre_path, gt_path=args.gt_path)