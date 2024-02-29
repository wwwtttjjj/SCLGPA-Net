import numpy as np
import os
from PIL import Image
num_classes = [0, 100, 150, 255]
NumClass = len(num_classes)
val_dice = np.zeros(NumClass)  # 初始化各类别的dice

def get_dice(y_pred, y_true):
    avg_dice = []  # 创建dice数组
    # 对每个类别求dice
    for i in num_classes:
        GT = y_true == i  # 获得该类别下的真实标签01分割图
        Pred = y_pred == i  # 获得该类别下的预测结果01分割图
        inter = np.sum(np.multiply(GT, Pred)) + 5e-08  # 求交集
        union = np.sum(GT) + np.sum(Pred) + 1e-07  # 求并集
        t = 2 * inter / union  # 计算dice
        avg_dice.append(t)  # 将该类别下的dice添加到数组中
    return avg_dice
if __name__ == '__main__':
    pseduo_paths = 'data/pseudo_label'
    label_paths = 'data/label'
    l = 0
    for f_name in os.listdir(label_paths):
        pseduo_path = os.path.join(pseduo_paths, f_name)
        label_path = os.path.join(label_paths, f_name)

        y_pseduo = np.array(Image.open(pseduo_path))
        y_label = np.array(Image.open(label_path))
        val_dice += np.array(get_dice(y_pseduo, y_label)) # 计算dice指标
        l += 1
    val_dice /= l
    print(val_dice)