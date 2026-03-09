import argparse, os
import torch
import torch.nn as nn

import numpy as np

from time import time

from utils import *

from model.TMBSN import TMBSN

from lmdb_loader import load_lmdb_infinite

from scipy.io import loadmat

import wandb

def main():
    parser = argparse.ArgumentParser(description="TM-BSN Training")
    parser.add_argument("--batchsize", type=int, default=4, help="Training batch size")
    parser.add_argument("--patchsize", type=int, default=128, help='training patch size')
    parser.add_argument("--dataset", type=str, default='SIDD', help='dataset SIDD or DND')
    parser.add_argument("--lmdb", type=str, required=True, help='path to lmdb directory')
    parser.add_argument("--print_every", type=int, default=1000, help='print step')
    parser.add_argument("--log_every", type=int, default=100, help='log step')
    parser.add_argument("--val_every", type=int, default=10000, help='validation step')
    parser.add_argument("--maxiter", type=int, default=500000, help="Number of training iterations")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--main_name", type=str, default='temp', help='main name of experiments')
    parser.add_argument("--sub_name", type=str, default='1st', help='sub name of experiments')
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument('--desc', type=str, default='', help='Experiment description for wandb logging')
    parser.add_argument("--h_set", type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='hole sizes for validation')

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
    val_imgs, val_blocks, _, _, _ = val_gt.shape  # (40, 32, 256, 256, 3)

    criterion_L1 = nn.L1Loss().to(device)

    BSN = TMBSN(in_channels=3, feat_channels=48, block_nums=5).to(device)
    print(f"Total BSN parameters: {sum(p.numel() for p in BSN.parameters())}")

    optimizer_BSN = torch.optim.Adam(BSN.parameters(), lr=float(args.lr))
    scheduler_BSN = CosineDecayScheduler(optimizer_BSN, float(args.lr), 200000, 300000, float(1e-6))

    BSN.train()

    best_psnr = {h: 0. for h in args.h_set}
    best_iter = {h: 0 for h in args.h_set}

    start_iter = 0

    if args.resume and os.path.isfile(args.resume):
        start_iter = load_checkpoint(args, BSN, optimizer_BSN, scheduler_BSN, device)

    t1 = time()

    for i, data in enumerate(train_dataset):
        cur_iter = start_iter + i

        noisy = data['noisy'].to(device)

        denoised = BSN(noisy, 5)

        loss_self = criterion_L1(denoised, noisy)
        loss_BSN = loss_self

        loss_BSN.backward()
        optimizer_BSN.step()
        optimizer_BSN.zero_grad()
        scheduler_BSN.step()

        if (cur_iter % args.log_every == 0) and (cur_iter != 0):
            wandb.log({"loss_BSN": loss_BSN, "loss_self": loss_self, "lr": optimizer_BSN.param_groups[0]['lr']}, step=cur_iter)

        if cur_iter % args.print_every == 0:
            print(f"[{cur_iter:06d}] BSN:{loss_BSN:.3f} self:{loss_self:.3f} time:{((time() - t1) / args.print_every):.3f}")
            t1 = time()

        if (cur_iter % args.val_every == 0) and (cur_iter != 0):
            BSN.eval()

            t2 = time()

            psnr_sums = {h: 0.0 for h in args.h_set}
            ssim_sums = {h: 0.0 for h in args.h_set}
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

                        results = eval_bsn_outs(val_noisy_img, val_gt_img, BSN, args.h_set, val_save_dir, count, pad=0)

                        for h in args.h_set:
                            psnr_sums[h] += results[h][0]
                            ssim_sums[h] += results[h][1]

            psnr_avgs = {h: psnr_sums[h] / count for h in args.h_set}
            ssim_avgs = {h: ssim_sums[h] / count for h in args.h_set}

            psnr_str = ' '.join(f'h{h}:{psnr_avgs[h]:.2f}' for h in args.h_set)
            ssim_str = ' '.join(f'h{h}:{ssim_avgs[h]:.3f}' for h in args.h_set)
            print(f"[{cur_iter:06d}] iterations val PSNR {psnr_str} SSIM {ssim_str}")

            ckpt_psnr_str = '_'.join(f'{psnr_avgs[h]:.2f}' for h in args.h_set)
            torch.save(BSN.state_dict(), os.path.join(save_dir_path, 'ckpt', f'bsn_{cur_iter}_{ckpt_psnr_str}.pth'))

            wandb.log({f"psnr_h{h}": psnr_avgs[h] for h in args.h_set}, step=cur_iter)
            wandb.log({f"ssim_h{h}": ssim_avgs[h] for h in args.h_set}, step=cur_iter)

            state = {
                'iter': cur_iter,
                'model': BSN.state_dict(),
                'optimizer': optimizer_BSN.state_dict(),
                'scheduler': scheduler_BSN.state_dict()
            }
            torch.save(state, os.path.join(save_dir_path, 'ckpt', 'bsn_latest.pth'))

            for h in args.h_set:
                if psnr_avgs[h] > best_psnr[h]:
                    best_psnr[h], best_iter[h] = psnr_avgs[h], cur_iter
                    torch.save(BSN.state_dict(), os.path.join(save_dir_path, 'ckpt', f'best_h{h}.pth'))

            for h in args.h_set:
                print(f'Best PSNR h{h}: {best_psnr[h]:.2f} at iter {best_iter[h]}')
            print(f'validation elapsed time: {(time() - t2):.0f} sec')

            BSN.train()
            t1 = time()

        if (cur_iter == args.maxiter):
            print('Finish!!')
            break


if __name__ == '__main__':
    main()
