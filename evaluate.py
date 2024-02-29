import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from medpy import metric
import numpy as np

def iou(matrix1, matrix2):
    intersection = np.logical_and(matrix1, matrix2)
    intersection_count = np.count_nonzero(intersection == 1)
    matrix1_count = np.count_nonzero(matrix1 == 1)
    matrix2_count = np.count_nonzero(matrix2 == 1)
    if matrix1_count == 0 and matrix2_count == 0:
        return 1
    iou = intersection_count / float(matrix1_count + matrix2_count - intersection_count)
    return iou

def calculate_hd95_asd(pred, gt):
    batch = pred.shape[0]
    now_batch = 0
    hd = 0
    asd = 0
    for i in range(batch):
        if len(np.unique(pred[i, ...])) == 1 or len(np.unique(gt[i, ...])) == 1:#其中一个全0不计算
            continue
        else:
            hd += metric.binary.hd95(pred[i, ...], gt[i, ...])
            asd += metric.binary.asd(pred[i, ...], gt[i, ...])
            now_batch += 1
    return hd, asd, now_batch


def evaluate(net, dataloader, device, num_classes):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score_P = 0
    dice_score_S = 0
    dice_score_I = 0
    hd95_asd_P_num, hd95_asd_S_num, hd95_asd_I_num = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    iou_a =[]
    # iou_srf =[]
    # iou_irf =[]


 
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long).squeeze(dim=1)
        mask_true = F.one_hot(mask_true, num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score_P += dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
            dice_score_S += dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)
            dice_score_I += dice_coeff(mask_pred[:, 3:, ...], mask_true[:, 3:, ...], reduce_batch_first=False)

            hd95_P, asd_P, num_P = calculate_hd95_asd(np.array(mask_pred[:, 1:2, ...].to('cpu')),np.array(mask_true[:, 1:2, ...].to('cpu')))
            hd95_asd_P_num[0] += hd95_P
            hd95_asd_P_num[1] += asd_P
            hd95_asd_P_num[2] += num_P
            hd95_S, asd_S, num_S = calculate_hd95_asd(np.array(mask_pred[:, 2:3, ...].to('cpu')),np.array(mask_true[:, 2:3, ...].to('cpu')))
            hd95_asd_S_num[0] += hd95_S
            hd95_asd_S_num[1] += asd_S          
            hd95_asd_S_num[2] += num_S            
            hd95_I, asd_I, num_I= calculate_hd95_asd(np.array(mask_pred[:, 3:, ...].to('cpu')),np.array(mask_true[:, 3:, ...].to('cpu')))
            hd95_asd_I_num[0] += hd95_I
            hd95_asd_I_num[1] += asd_I
            hd95_asd_I_num[2] += num_I

            iou_a.append(iou(np.array(mask_pred[:, 1:2, ...].to('cpu')),np.array(mask_true[:, 1:2, ...].to('cpu'))))
            iou_a.append(iou(np.array(mask_pred[:, 2:3, ...].to('cpu')),np.array(mask_true[:, 2:3, ...].to('cpu'))))
            iou_a.append(iou(np.array(mask_pred[:, 3:, ...].to('cpu')),np.array(mask_true[:, 3:, ...].to('cpu'))))




    net.train()
    hd95_score = ((hd95_asd_P_num[0] / (hd95_asd_P_num[2] + 1e-10)) + (hd95_asd_S_num[0] / (hd95_asd_S_num[2] + 1e-10)) + (hd95_asd_I_num[0] / (hd95_asd_I_num[2] + 1e-10))) / 3
    asd_score = ((hd95_asd_P_num[1] / (hd95_asd_P_num[2] + 1e-10))+ (hd95_asd_S_num[1] / (hd95_asd_S_num[2] + 1e-10))+ (hd95_asd_I_num[1] / (hd95_asd_I_num[2] + 1e-10))) / 3
    # PDE_i = np.mean(np.delete(iou_ped, np.where(iou_ped==1)))
    # SRF_i = np.mean(np.delete(iou_srf, np.where(iou_srf==1)))
    # IRF_i = np.mean(np.delete(iou_irf, np.where(iou_irf==1)))
    # std_p = np.std(np.delete(iou_ped, np.where(iou_ped==1)))
    # # std_s = np.std(np.delete(iou_srf, np.where(iou_srf==1)))
    # # std_i = np.std(np.delete(iou_irf, np.where(iou_irf==1)))
    # print(iou_ped)
    # print(iou_srf)
    # print(iou_irf)
    # iou_ped = [x for x in iou_ped if x != 0]
    # iou_srf = [x for x in iou_srf if x != 0]
    iou_a = [x for x in iou_a if x != 0 and x != 1]







    return dice_score_P / num_val_batches, dice_score_S / num_val_batches, dice_score_I / num_val_batches, asd_score, hd95_score, np.mean(iou_a), np.std(iou_a)