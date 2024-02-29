from cleanlab.filter import find_label_issues
import numpy as np
import torch
import random

#考虑内存是不是放不下，要不一个一个从显卡拿，但这样会慢很多
# def get_masks(pred_labels, weak_masks, p_maps, device):
#     labels, pred_probs, p_maps = np.array(weak_masks.detach().cpu()), np.array(
#         pred_labels.detach().cpu()), np.array(p_maps)
#     N, C, H, W = pred_probs.shape
#     rst = []
#     for batch in range(N):
#         cf_pmap = p_maps[batch,:,:].reshape(H * W)
#         cf_label = labels[batch,:,:].reshape(H * W)
#         # if len(np.unique(cf_label)) == 1:#如果概率图谱都大于阈值，不需要自信学习，直接信任概率图谱
#         #     rst.append(labels[batch,:,:])
#         #     continue
#         cf_pred = np.squeeze(pred_probs[batch,:,:,:]).reshape(-1, H * W).T
#         # print(cf_label.shape, cf_pred.shape, np.unique(cf_label))
#         error_maps = confinence_learning(cf_label, cf_pred)
#         error_maps = update_errormaps(list(error_maps),cf_pmap)  #根据p_maps和error_maps更新error_maps
#         cf_weak_mask = label_refinement(error_maps, cf_label, cf_pred)  #label_refinement
#         rst.append(cf_weak_mask.reshape(H, W))
#     cl_weak_masks = torch.tensor(np.stack(rst)).to(device, dtype = torch.long)
#     return cl_weak_masks

def get_masks(pred_labels, weak_masks, p_maps, device, cl = False):
    if cl == False:
        return weak_masks, p_maps.to(device=device, dtype=torch.float32)
    N, C, H, W = pred_labels.shape
    rst = []
    p_rst = []
    for batch in range(N):
        label, pred_prob = np.array(weak_masks[batch].detach().cpu()), np.array(
        pred_labels[batch].detach().cpu())
        p_map = p_maps[batch]

        cf_label = label.reshape(H * W)
        p_map = p_map.reshape(H * W)
        cf_pred = np.squeeze(pred_prob).reshape(-1, H * W).T
        # print(cf_label.shape, cf_pred.shape, np.unique(cf_label))
        error_maps = confinence_learning(cf_label, cf_pred)
        cf_weak_mask,cf_map = label_refinement(error_maps, cf_label, cf_pred, p_map)  #label_refinement
        # print(len(error_maps))
        rst.append(cf_weak_mask.reshape(H, W))
        p_rst.append(cf_map.reshape(H, W))
    cl_weak_masks = torch.tensor(np.stack(rst)).to(device, dtype = torch.long)
    cl_maps = torch.tensor(np.stack(p_rst)).to(device, dtype = torch.float32)
    return cl_weak_masks, cl_maps

#自信学习
def confinence_learning(labels, pred_probs):
    ordered_label_issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,  # out-of-sample predicted probabilities from any model
        return_indices_ranked_by='self_confidence',
        min_examples_per_class = 0,
        # filter_by = 'both'
    )
    return ordered_label_issues

# #根据概率图谱更新错误的坐标，只取小于图谱阈值的坐标
# def update_errormaps(error_maps, p_maps):
#     p_maps = p_maps.flatten()
#     error_maps = [i for i in range(len(error_maps)) if p_maps[i] <= 0.5]
#     return np.array(error_maps)

#标签修复，把自信学习后且过滤掉的错误的坐标更新为模型预测的坐标
def label_refinement(error_map, label, pred_prob,p_map):
    for error in error_map:
        if p_map[error] >= 0.8:
            continue
        else:
            label[error] = np.argmax(pred_prob[error])
            p_map[error] = 1
    return label, p_map


# def get_masks_onlyby_pro(pred_labels, weak_masks, p_maps, device):
#     labels, pred_probs, p_maps = np.array(weak_masks.detach().cpu()), np.array(
#     pred_labels.detach().cpu()), np.array(p_maps)

#     pred_probs = pred_probs.transpose((0,2,3,1))
#     changed_label = np.where(p_maps <= 0.7)#把pro_map里所有0.5以下的都给更为mask_pred的值
#     labels[changed_label] = np.argmax(pred_probs[changed_label], axis=1)
#     cl_weak_masks = torch.tensor(labels).to(device, dtype = torch.long)
#     return cl_weak_masks
