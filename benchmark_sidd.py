import argparse, os
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from utils import *

from model.NBSN102 import N_BSN
from model.TMBSN import TMBSN

from scipy.io import loadmat

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="SIDD Benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--name", type=str, default='temp', help='output directory name')
    parser.add_argument("--ckpt", type=str, required=True, help='checkpoint path')
    parser.add_argument("--model", type=str, required=True, choices=['tmbsn', 'nbsn'], help='model selection')
    parser.add_argument("--pad", type=int, default=16, help='padding size')
    parser.add_argument("--h", type=int, default=2, help='hole size (TMBSN only)')

    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint doesn't exist: {args.ckpt}")

    save_dir_path = os.path.join('./benchmark', args.name)
    os.makedirs(save_dir_path, exist_ok=True)

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

    val_noisy = np.array(loadmat('./dataset/BenchmarkNoisyBlocksSrgb.mat')['BenchmarkNoisyBlocksSrgb'], dtype=np.float32) / 255.0
    val_imgs, val_blocks, val_h, val_w, val_c = val_noisy.shape

    p = args.pad

    DN.eval()

    output_blocks_base64string = []

    with torch.no_grad():
        for i in tqdm(range(val_imgs)):
            for j in tqdm(range(val_blocks), leave=False):
                noisy = val_noisy[i, j:j+1, :, :, :]

                noisy = (torch.from_numpy(noisy).permute(0, 3, 1, 2)).to(device)

                noisy = F.pad(noisy, (p, p, p, p), mode="reflect") if p != 0 else noisy

                if args.model == 'tmbsn':
                    denoised = DN(noisy, args.h)
                else:
                    denoised = DN(noisy)

                denoised = denoised[..., p:-p, p:-p] if p != 0 else denoised

                out_block = np.clip(denoised.permute(0, 2, 3, 1).cpu().numpy(), 0.0, 1.0)
                out_block = np.round(out_block[0] * 255).astype(np.uint8)

                out_block_base64string = array_to_base64string(out_block)
                output_blocks_base64string.append(out_block_base64string)

    output_file_name = 'SubmitSrgb.csv'
    output_file_path = os.path.join(save_dir_path, output_file_name)
    print(f'Saving outputs to {output_file_path}')
    output_df = pd.DataFrame()
    n_blocks = len(output_blocks_base64string)
    print(f'Number of blocks = {n_blocks}')
    output_df['ID'] = np.arange(n_blocks)
    output_df['BLOCK'] = output_blocks_base64string

    output_df.to_csv(output_file_path, index=False)

    print('Done.')


if __name__ == '__main__':
    main()
