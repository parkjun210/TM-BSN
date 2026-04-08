import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from time import time

from utils import *

from model.TMBSN import TMBSN
from model.NBSN102 import N_BSN

from lmdb_loader import load_lmdb_infinite

from scipy.io import loadmat

import cv2

import wandb

def main():
    parser = argparse.ArgumentParser(description="TM-BSN Knowledge Distillation Training")
    parser.add_argument("--batchsize", type=int, default=8, help="Training batch size")
    parser.add_argument("--patchsize", type=int, default=128, help='training patch size')
    parser.add_argument("--dataset", type=str, default='SIDD', help='dataset SIDD or DND')
    parser.add_argument("--lmdb", type=str, required=True, help='path to lmdb directory')
    parser.add_argument("--print_every", type=int, default=1000, help='print step')
    parser.add_argument("--log_every", type=int, default=100, help='log step')
    parser.add_argument("--val_every", type=int, default=10000, help='validation step')
    parser.add_argument("--maxiter", type=int, default=200000, help="Number of training iterations")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--main_name", type=str, default='temp', help='main name of experiments')
    parser.add_argument("--sub_name", type=str, default='1st', help='sub name of experiments')
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument('--desc', type=str, default='', help='Experiment description for wandb logging')

    # Distillation args
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to pretrained TMBSN checkpoint")
    parser.add_argument("--pad", type=int, default=16, help='padding size for teacher input')
    parser.add_argument("--h_set", type=int, nargs='+', default=[2, 3, 4, 5, 6], help='hole sizes for teacher forward_outs')
    parser.add_argument("--recharge", action="store_true", help="Apply Recharger to teacher outputs; if not set, use raw teacher outputs as targets")

    args = parser.parse_args()

    save_dir_path = make_indexed_dir(args.main_name, args.sub_name, root_dir='./output')

    with open(os.path.join(save_dir_path, 'model.txt'), 'w') as f:
        f.write(str(args._get_kwargs()))
    cpy_code(os.path.join(save_dir_path, 'codes'))

    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login()
    wandb.init(
        project=args.main_name,
        name=os.path.basename(save_dir_path),
        dir=save_dir_path,
        config=vars(args),
        notes=args.desc,
        settings=wandb.Settings(_service_wait=60)
    )

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    seed = int(time()) % 10000
    set_random_seed(seed)

    train_dataset = load_lmdb_infinite(args.lmdb, args.patchsize, args.batchsize)

    val_noisy = np.array(loadmat('./dataset/ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb'], dtype=np.float32) / 255.0
    val_gt = np.array(loadmat('./dataset/ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb'], dtype=np.float32) / 255.0
    val_imgs, val_blocks, val_h, val_w, val_c = val_gt.shape  # (40, 32, 256, 256, 3)

    criterion_L1 = nn.L1Loss().to(device)

    # Teacher model
    BSN = TMBSN(in_channels=3, feat_channels=48, block_nums=5).to(device)
    BSN.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    BSN.eval()

    # Student model
    NBD = N_BSN().to(device)

    print(f"Total BSN parameters: {sum(p.numel() for p in BSN.parameters())}")
    print(f"Total NBD parameters: {sum(p.numel() for p in NBD.parameters())}")

    optimizer_NBD = torch.optim.Adam(NBD.parameters(), lr=float(args.lr))
    scheduler_NBD = CosineDecayScheduler(optimizer_NBD, float(args.lr), 100000, 100000, float(1e-6))

    NBD.train()

    if args.recharge:
        recharger = Recharger()
        print('Recharger enabled')
    else:
        print('Recharger disabled — using raw teacher outputs as targets')

    p = args.pad
    h_set = args.h_set
    print(f'h_set: {h_set}')

    best_nbd = 0.
    best_nbd_iter = 0

    start_iter = 0

    if args.resume and os.path.isfile(args.resume):
        start_iter = load_checkpoint(args, NBD, optimizer_NBD, scheduler_NBD, device)

    t1 = time()

    for i, data in enumerate(train_dataset):
        cur_iter = start_iter + i

        noisy = data['noisy'].to(device)

        denoised = NBD(noisy)

        noisy_padded = F.pad(noisy, (p, p, p, p), mode="reflect") if p != 0 else noisy

        with torch.no_grad():
            targets = BSN.forward_outs(noisy_padded, h_set)

            if p != 0:
                targets = [t[..., p:-p, p:-p] for t in targets]
                noisy_padded = noisy_padded[..., p:-p, p:-p]

            if args.recharge:
                t_rd_elements = recharger.apply(noisy_padded, targets)
            else:
                t_rd_elements = targets

        loss_NBD = sum(criterion_L1(denoised, T_RD) for T_RD in t_rd_elements)

        loss_NBD.backward()
        optimizer_NBD.step()
        optimizer_NBD.zero_grad()
        scheduler_NBD.step()

        if (cur_iter % args.log_every == 0) and (cur_iter != 0):
            wandb.log({"loss_NBD": loss_NBD, "lr": optimizer_NBD.param_groups[0]['lr']}, step=cur_iter)

        if cur_iter % args.print_every == 0:
            print(f"[{cur_iter:06d}] NBD:{loss_NBD:.3f} time:{((time() - t1) / args.print_every):.3f}")
            t1 = time()

        if (cur_iter % args.val_every == 0) and (cur_iter != 0):
            NBD.eval()

            t2 = time()

            psnr_nbd_sum = 0.0
            ssim_nbd_sum = 0.0
            count = 0

            val_save_dir = os.path.join(save_dir_path, 'val_images', str(cur_iter))
            os.makedirs(val_save_dir, exist_ok=True)

            with torch.no_grad():
                for i in range(val_imgs):
                    for j in range(val_blocks):
                        val_noisy_img = val_noisy[i, j:j+1, :, :, :]
                        val_gt_img = val_gt[i, j:j+1, :, :, :]

                        val_noisy_img = (torch.from_numpy(val_noisy_img).permute(0, 3, 1, 2)).to(device)

                        count += 1

                        psnr_nbd, ssim_nbd = eval_nbd_img(val_noisy_img, val_gt_img, NBD, val_save_dir, count, 'tmbsnRD', pad=16)

                        psnr_nbd_sum += psnr_nbd
                        ssim_nbd_sum += ssim_nbd

            psnr_avg = psnr_nbd_sum / count
            ssim_avg = ssim_nbd_sum / count
            print(f"[{cur_iter:06d}] iterations val PSNR:{psnr_avg:.2f} SSIM:{ssim_avg:.3f}")
            torch.save(NBD.state_dict(), os.path.join(save_dir_path, 'ckpt', f'nbd_{cur_iter}_{psnr_avg:.2f}.pth'))
            wandb.log({"psnr_avg": psnr_avg, "ssim_avg": ssim_avg}, step=cur_iter)

            state = {
                'iter': cur_iter,
                'model': NBD.state_dict(),
                'optimizer': optimizer_NBD.state_dict(),
                'scheduler': scheduler_NBD.state_dict()
            }
            torch.save(state, os.path.join(save_dir_path, 'ckpt', 'nbd_latest.pth'))

            if psnr_avg > best_nbd:
                best_nbd, best_nbd_iter = psnr_avg, cur_iter
                torch.save(NBD.state_dict(), os.path.join(save_dir_path, 'ckpt', 'best_nbd.pth'))

            print(f'Best PSNR NBD: {best_nbd:.2f} at iter {best_nbd_iter}')
            print(f'validation elapsed time: {(time() - t2):.0f} sec')

            NBD.train()
            t1 = time()

        if (cur_iter == args.maxiter):
            print('Finish!!')
            break


if __name__ == '__main__':
    main()
