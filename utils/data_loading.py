from os import listdir
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

# from utils.dataGenerator import sample_random_patch_png, sample_label_with_coor_png
color2label = [100, 150, 255]

# 将文件夹里的数据读成dataset
class BasicDataset(Dataset):
    # 构造函数
    def __init__(self,
                 imgs_dir,
                 masks_dir,
                 augumentation,
                 probability_dir=None,
                 ema_trans = None):
        self.imgs_dir = imgs_dir  # 图像文件
        self.masks_dir = masks_dir  # 标签文件夹
        self.probability_dir = probability_dir
        self.augumentation = augumentation
        self.ema_trans = ema_trans


        # 获得所有图像文件名
        self.ids = [
            file for file in listdir(imgs_dir) if not file.startswith('.')
        ]

    # dataset长度函数
    def __len__(self):
        return len(self.ids)

    # dataset返回对应位置数据函数
    def __getitem__(self, i):
        idx = self.ids[i]  # 获取第i个图像的文件名
        mask_file = self.masks_dir + idx  # 获取第i个标签的文件路径
        img_file = self.imgs_dir + idx  # 获取第i个图像的文件路径

        mask_original = np.asarray(Image.open(mask_file))  # 读取标签
        mask = mask_original.copy()  # 有时候没有修改权限，需要复制一份
        for i in range(len(color2label)):
            mask[mask==color2label[i]] = i + 1

        image = cv2.imread(img_file,0)

        if self.probability_dir:
            npy_file = np.load(self.probability_dir + idx[:-4] + '.npy')
            probability_map = npy_file.copy()
            probability_map = probability_map / 10  #npy file保存的概率，值在0-10之间
            mask_pro = np.stack((mask, probability_map)).transpose(1,2,0)
            augment = self.augumentation(image = image, mask = mask_pro)
            image, mask_pro = augment['image'], augment['mask'][None]
            mask_pro = mask_pro.squeeze(dim = 0).permute(2,0,1)
            mask, probability_map = mask_pro[0], mask_pro[1]
            return {
                'image':image / 255.,  # 返回tensor类型的图像
                'mask':mask,  # 返回tensor类型的标签
                'probability_map':probability_map#返回概率图谱
                # 'idx':idx[:-4] + '.png'
            }
        else:
            augment = self.augumentation(image = image, mask = mask)
            image, mask = augment['image'], augment['mask'][None]
            return {
                'image':image / 255.,  # 返回tensor类型的图像
                'mask':mask.squeeze(dim = 0)  # 返回tensor类型的标签
                # 'idx':idx[:-4] + '.png'
            }
