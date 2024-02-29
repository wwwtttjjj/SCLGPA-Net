import numpy as np
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import data_loading2
from functions import create_model
from transformers import transformer_img
import get_errormaps_2
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_path',
                        type=str,
                        default='semi-data3/weak_labeled_data/',
                        help='the path of training data')

    parser.add_argument('--model_path',
                        type=str,
                        default='checkpoints/SGPA.pth',
                        help='the path of save_model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')

    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='the batch_size of training size')

    args = parser.parse_args()
    return args
label2mask = [100, 150, 255]

if __name__ == "__main__":
    args = get_parser()

    weak_path = args.weak_path
    save_path = weak_path + 'masks_cl'
    map_path = weak_path + 'probability_cl'
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    num_classes = 4
    weak_dataset = data_loading2.BasicDataset(
        imgs_dir=weak_path + '/' + 'imgs/',
        masks_dir=weak_path + '/' + 'masks/',
        probability_dir = weak_path + '/' + 'probability_maps/',
        augumentation=transformer_img())
    weak_dataloader = DataLoader(dataset=weak_dataset,
                                batch_size=batch_size,#weak labeled data
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                               )
    total = len(weak_dataloader)

    cl_model = create_model(device=device,num_classes=num_classes, ema=True)
    cl_model.load_state_dict(torch.load(args.model_path, map_location=device))

    cl_model.eval()


    for i, sampled_batch in enumerate(weak_dataloader):
        weak_imgs, weak_masks, p_maps, idx = sampled_batch['image'], sampled_batch['mask'], sampled_batch['probability_map'], sampled_batch['idx']

        weak_imgs, weak_masks = weak_imgs.to(device=device, dtype=torch.float32), weak_masks.to(
                device=device, dtype=torch.long)

        ema_inputs = weak_imgs
        with torch.no_grad():
            ema_outputs = cl_model(ema_inputs)
            pred_labels = F.softmax(ema_outputs, dim=1).float()
            cl_weak_masks,p_maps_ce =  get_errormaps_2.get_masks(pred_labels, weak_masks, p_maps, device, cl = True)
            cl_weak_masks,p_maps_ce = cl_weak_masks.cpu(),np.array(p_maps_ce.cpu())
            N, H, W = cl_weak_masks.shape
            for n in range(N):
                cl_masks = cl_weak_masks[n]
                p_map = p_maps[n]
                for j in range(len(label2mask)):
                    cl_masks[cl_masks == (j + 1)] = label2mask[j]
                cl_masks = Image.fromarray(np.uint8(np.squeeze(cl_masks)))
                # cl_masks = cl_masks.resize((600, 250), Image.NEAREST)
                cl_masks.save(save_path + '/' + idx[n])
                
                p_map = Image.fromarray(np.uint8(np.squeeze(p_map*10)))
                # p_map = p_map.resize((600, 250), Image.NEAREST)
                np.save(map_path + '/' + idx[n][:-4] + '.npy', p_map)
        print(i / len(weak_dataloader), end = '')
