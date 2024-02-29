import os
import shutil
import random

def save_img_mask(source_dirs, target_imgs, target_masks):
    i = 0
    j = 0

    # 遍历源目录
    for source_dir in source_dirs:
        masks_dir = os.path.join(source_dir, 'masks')
        imgs_dir = os.path.join(source_dir, 'imgs')
        
        # 遍历masks目录
        for filename in os.listdir(masks_dir):
            source_path = os.path.join(masks_dir, filename)
            target_path = os.path.join(target_masks, f'slice_{i}.png')
            shutil.copy(source_path, target_path)
            i += 1
        
        # 遍历imgs目录
        for filename in os.listdir(imgs_dir):
            source_path = os.path.join(imgs_dir, filename)
            target_path = os.path.join(target_imgs, f'slice_{j}.png')
            shutil.copy(source_path, target_path)
            j+=1
def split_val_test():
    sort_volume = [i for i in range(14,24)]
# random.shuffle(sort_volume)
    # 源目录
    val_dirs = ['val+test/'+f'{sort_volume[i]}' for i in range(5)]
    test_dirs = ['val+test/'+f'{sort_volume[i]}' for i in range(5, 10)]
    # 目标目录

    test_imgs = 'test/imgs'
    test_masks = 'test/masks'
    val_imgs = 'val/imgs'
    val_masks = 'val/masks'
    save_img_mask(val_dirs, val_imgs, val_masks)
    save_img_mask(test_dirs, test_imgs, test_masks)

if __name__ =='__main__':
    print('1111111111')
