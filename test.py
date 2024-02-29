from evaluate import evaluate
from transformers import transformer_img, val_form, height, width
from unet import UNet
import argparse
import torch
from utils import data_loading
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath',
                    type=str,
                    default='./checkpoints/SGPLG.pth',
                    help='the path of val data')
args = parser.parse_args()
val_path = 'test/'
val_dataset = data_loading.BasicDataset(
    imgs_dir=val_path + '/' + 'imgs/',
    masks_dir=val_path + '/' + 'masks/',
    augumentation=val_form())
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_classes = 4
model = UNet(n_channels=1, n_classes=4, bilinear=False)
model.load_state_dict(torch.load(args.modelpath, map_location='cpu'))
model.to(device)
val_P, val_S, val_I, asd_score, hd95_score,m_iou,std_iou = evaluate(model, val_dataloader, device, num_classes)

val_dice = (val_P + val_S + val_I) / 3
print('val_dice:{}, val_S:{},val_I:{},val_P:{} ,asd:{},hd95:{},m_iou:{},std,{}'
      .format(round(val_dice.item(),4), round(val_S.item(), 4), round(val_I.item(),4), round(val_P.item(), 4), round(asd_score.item(),2), 
              round(hd95_score.item(),2),m_iou, std_iou))
