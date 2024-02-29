from utils import ramps
from unet import UNet
import os
import torch
import logging
def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,
                                                   args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):

    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def create_model(device, num_classes = 4, ema=False):
    # Network definition
    net = UNet(n_channels=1, n_classes=num_classes)
    model = net.to(device)
    #截断反向传播的梯度流
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def save_model(args, global_step, model):
        torch.save(model.state_dict(), 'checkpoints/SGPLG.pth')
        print(f'Checkpoint {global_step} saved!')

