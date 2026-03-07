import argparse, os
import torch
import torch.nn.functional as F
import numpy as np

from utils import *

from model.TMBSN import TMBSN
from model.NBSN102 import N_BSN

from scipy.io import loadmat

import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="SIDD Validation")
    parser.add_argument("--ckpt", type=str, required=True, help='checkpoint path')
    parser.add_argument("--model", type=str, required=True, choices=['tmbsn', 'nbsn'], help='model selection')
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--pad", type=int, default=0, help='padding for inference')
    parser.add_argument("--name", type=str, default='validation', help='output directory name')
    parser.add_argument("--h_set", type=int, nargs='+', default=[1, 2], help='hole sizes for evaluation (TMBSN only)')

    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint doesn't exist: {args.ckpt}")

    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'tmbsn':
        DN = TMBSN(in_channels=3, feat_channels=48, block_nums=5).to(device)
    else:
        DN = N_BSN().to(device)

    print(f"Total parameters: {sum(p.numel() for p in DN.parameters())}")

    DN.load_state_dict(torch.load(args.ckpt, map_location=device))
    DN.eval()

    val_noisy = np.array(loadmat('./dataset/ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb'], dtype=np.float32) / 255.0
    val_gt = np.array(loadmat('./dataset/ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb'], dtype=np.float32) / 255.0
    val_imgs, val_blocks, val_h, val_w, val_c = val_gt.shape  # (40, 32, 256, 256, 3)

    save_dir = os.path.join('./validation', args.name)
    os.makedirs(save_dir, exist_ok=True)

    count = 0

    if args.model == 'tmbsn':
        psnr_sums = {h: 0.0 for h in args.h_set}
        ssim_sums = {h: 0.0 for h in args.h_set}

        with torch.no_grad():
            for i in tqdm(range(val_imgs)):
                for j in tqdm(range(val_blocks), leave=False):
                    val_noisy_img = val_noisy[i, j:j+1, :, :, :]
                    val_gt_img = val_gt[i, j:j+1, :, :, :]

                    val_noisy_img = (torch.from_numpy(val_noisy_img).permute(0, 3, 1, 2)).to(device)

                    count += 1

                    results = eval_bsn_outs(val_noisy_img, val_gt_img, DN, args.h_set, save_dir, count, pad=args.pad)

                    for h in args.h_set:
                        psnr_sums[h] += results[h][0]
                        ssim_sums[h] += results[h][1]

        for h in args.h_set:
            psnr_avg = psnr_sums[h] / count
            ssim_avg = ssim_sums[h] / count
            print(f"SIDD Validation h{h} PSNR: {psnr_avg:.2f} SSIM: {ssim_avg:.4f}")

    else:
        psnr_sum = 0.0
        ssim_sum = 0.0

        with torch.no_grad():
            for i in tqdm(range(val_imgs)):
                for j in tqdm(range(val_blocks), leave=False):
                    val_noisy_img = val_noisy[i, j:j+1, :, :, :]
                    val_gt_img = val_gt[i, j:j+1, :, :, :]

                    val_noisy_img = (torch.from_numpy(val_noisy_img).permute(0, 3, 1, 2)).to(device)

                    count += 1

                    psnr_v, ssim_v = eval_nbd_img(val_noisy_img, val_gt_img, DN, save_dir, count, 'nbsn', pad=args.pad)

                    psnr_sum += psnr_v
                    ssim_sum += ssim_v

        psnr_avg = psnr_sum / count
        ssim_avg = ssim_sum / count
        print(f"SIDD Validation PSNR: {psnr_avg:.2f} SSIM: {ssim_avg:.4f}")


if __name__ == '__main__':
    main()
