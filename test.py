import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
import time
from model.Network import SAANet
from utils.data import test_dataset

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352,
                    help='testing size')
opt = parser.parse_args()

dataset_path = './datasets/'
test_datasets = ['EORSSD']

model = SAANet()
model.load_state_dict(torch.load('./models/EORSSD/SAANet_EORSSD.pth.59'))
model.cuda()
model.eval()

for dataset in test_datasets:
    save_path = f'./results_EORSSD/{dataset}/'
    os.makedirs(save_path, exist_ok=True)

    image_root = f'{dataset_path}{dataset}/val/val-images/'
    gt_root = f'{dataset_path}{dataset}/val/val-labels/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.cuda()
            start_time = time.time()

            res, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, x1_edge, x2_edge, x3_edge, x4_edge = model(image)

            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            imageio.imwrite(f'{save_path}{name}', (res * 255).astype(np.uint8))

            if i == test_loader.size - 1:
                time_sum += time.time() - start_time
                print(f'Average speed: {test_loader.size / time_sum:.4f} fps')