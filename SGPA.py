import os
import random
import numpy as np
from itertools import cycle
from tqdm import tqdm
import warnings
import logging


warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from get_args import get_parser
from utils import losses, dice_score, data_loading
from functions import update_ema_variables, create_model
from evaluate import evaluate
from transformers import transformer_img, val_form
from random_val_test import split_val_test

import os
os.environ["WANDB_MODE"] = "dryrun"

def save_model(model):
        torch.save(model.state_dict(), 'checkpoints/SGPA.pth')
        print(f'Checkpoint {global_step} saved!')

if __name__ == "__main__":
    args = get_parser()
    split_val_test()

    train_path = args.train_path
    labeled_path = train_path + '/' + "labeled_data"
    weak_path = train_path + '/' + "weak_labeled_data"
    val_path = args.val_path
    test_path = args.test_path

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    amp = args.amp
    num_classes = 4

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')


    labeled_dataset = data_loading.BasicDataset(
        imgs_dir=labeled_path + '/' + 'imgs/',
        masks_dir=labeled_path + '/' + 'masks/',
        augumentation=transformer_img())

    weak_dataset = data_loading.BasicDataset(
        imgs_dir=weak_path + '/' + 'imgs/',
        masks_dir=weak_path + '/' + 'masks/',
        probability_dir=weak_path + '/' + 'probability_maps/',
        augumentation=transformer_img()
        )
    n_train = len(weak_dataset)

    val_dataset = data_loading.BasicDataset(
        imgs_dir=val_path + '/' + 'imgs/',
        masks_dir=val_path + '/' + 'masks/',
        augumentation=val_form())
    test_dataset = data_loading.BasicDataset(
        imgs_dir=test_path + '/' + 'imgs/',
        masks_dir=test_path + '/' + 'masks/',
        augumentation=val_form())
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                    batch_size=2,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    worker_init_fn=worker_init_fn)
    weak_dataloader = DataLoader(dataset=weak_dataset,
                                batch_size=4,#weak labeled data
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
                                
    '''wandb'''# (Initialize logging)

    logging.info(f'''Starting training:
    Epochs:          {epochs}
    Batch size(labeled, weak_labeled):      {2, 4}
    Learning rate:   {learning_rate}
    Checkpoints:     {args.save_path}
    Device:          {device.type}
    Mixed Precision: {amp}
    ''')

    '''model'''
    model = create_model(device=device,num_classes=num_classes)
    # model.load_state_dict(torch.load(r'C:\Users\10194\Desktop\Unet_effusion_segmentation\checkpoints\test_100.pth', map_location=torch.device('cuda')))

    ema_model = create_model(device=device,num_classes=num_classes, ema=True)
    model.train()
    ema_model.train()
    '''some para'''
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    consistency_criterion = losses.softmax_mse_loss
    CEloss = torch.nn.CrossEntropyLoss(reduction='none')
    global_step = 0
    max_iou=0

    for epoch in range(1, epochs+1):
        model.train()
        with tqdm(total=(n_train)/4, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i, sampled_batch in enumerate(
                    zip(cycle(labeled_dataloader),weak_dataloader)):
                #labeled data and weak labeded data
                labeled_imgs, labeled_masks = sampled_batch[0]['image'], sampled_batch[0]['mask']
                labeled_imgs, labeled_masks = labeled_imgs.to(device=device, dtype=torch.float32), labeled_masks.to(
                        device=device, dtype=torch.long)
                weak_imgs, weak_masks, p_maps = sampled_batch[1]['image'], sampled_batch[1]['mask'], sampled_batch[1]['probability_map']
                weak_imgs, weak_masks, p_maps_CE = weak_imgs.to(device=device, dtype=torch.float32), weak_masks.to(
                            device=device, dtype=torch.long),p_maps.to(device=device, dtype=torch.float32)
                #噪声
                noise = torch.clamp(torch.randn_like(weak_imgs) * 0.05, -0.05,0.05)
                ema_inputs = weak_imgs + noise

                #前向传播（model and emamodel）
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs_labeled = model(labeled_imgs)
                    #计算有监督损失
                    supervised_loss = torch.mean(CEloss(
                        outputs_labeled, labeled_masks)) + dice_score.dice_loss(
                            F.softmax(outputs_labeled, dim=1).float(),
                            F.one_hot(labeled_masks, num_classes).permute(0, 3, 1,
                                                                        2).float(),
                            multiclass=True)
                    supervised_loss = args.alpha * supervised_loss
                    outputs_weak = model(weak_imgs)
                    # with torch.no_grad():
                    #     ema_outputs = ema_model(ema_inputs)
                    # consistency_weight = get_current_consistency_weight(args, (global_step - 2000) //
                    #                                                     600)
                    # consistency_dist = consistency_criterion(outputs_weak, ema_outputs)
                    # consistency_loss = consistency_weight * consistency_dist / (batch_size)
                    consistency_loss = 0
                    #计算弱监督损失
                    if global_step>=300:
                        outputs_weak = model(weak_imgs)
                        weak_supervised_loss = torch.mean(CEloss(outputs_weak, weak_masks) * p_maps_CE) + dice_score.dice_loss(
                                F.softmax(outputs_weak, dim=1).float(),
                                F.one_hot(weak_masks, num_classes).permute(0, 3, 1,
                                                                            2).float(),
                                multiclass=True)
                        weak_supervised_loss = args.beta * weak_supervised_loss
                    else:
                        weak_supervised_loss = 0
                    #batch 的总共的loss
                    loss = supervised_loss + weak_supervised_loss + consistency_loss


                #student model反向传播
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)


                grad_scaler.update()
                pbar.update(labeled_imgs.shape[0])
                global_step += 1
                #teacher model EMA更新参数
                update_ema_variables(model, ema_model, args.ema_decay, global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                #评估,每个epoch评估2次
                division_step = (n_train // (2 * batch_size))
                if global_step != 0 and global_step % 100 == 0:
                    val_P, val_S, val_I, asd_score, hd95_score, m_iou, std_iou = evaluate(model, val_dataloader, device, num_classes)
                    val_dice = (val_P + val_S + val_I) / 3
                    if m_iou > max_iou:
                        save_model(model)#save best model
                        max_iou = m_iou
                    logging.info('Validation Dice score: {},{},{},{},{},{}'.format(val_P, val_S, val_I, val_dice, m_iou,std_iou))

                #2k降lr倍数0.1
                if global_step % 4000 == 0:
                    lr = learning_rate * 0.1 ** (global_step // 4000)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

    model_test = create_model(device=device,num_classes=num_classes)
    model_test.load_state_dict(torch.load('checkpoints/SGPA.pth', map_location='cpu'))
    model_test.to(device)
    test_P, test_S, test_I, asd_score, hd95_score,m_iou,std_iou = evaluate(model_test, test_dataloader, device, num_classes)
    test_dice = (test_P + test_S + test_I) / 3
    with open('SGPA.txt', 'a') as file:
        # 写入文本内容，这将添加到文件的末尾
        file.write('test Iou score: {},{}\n'.format(round(m_iou, 4),round(std_iou,4)))
    logging.info('test Iou score: {},{}\n'.format(round(m_iou, 4),round(std_iou,4)))

