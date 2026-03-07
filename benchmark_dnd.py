import argparse, os
import torch
import torch.nn.functional as F

import numpy as np

from utils import *

from model.NBSN102 import N_BSN
from model.TMBSN import TMBSN

import scipy.io as sio

from tqdm import tqdm
import h5py


def main():
    parser = argparse.ArgumentParser(description="DND Benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--name", type=str, default='temp', help='output directory name')
    parser.add_argument("--ckpt", type=str, required=True, help='checkpoint path')
    parser.add_argument("--model", type=str, required=True, choices=['tmbsn', 'nbsn'], help='model selection')
    parser.add_argument("--pad", type=int, default=0, help='padding size')
    parser.add_argument("--h", type=int, default=2, help='hole size (TMBSN only)')

    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint doesn't exist: {args.ckpt}")

    save_dir_path = os.path.join('./benchmark', args.name, 'mat_pieces')
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

    DN.eval()

    p = args.pad

    data_folder = './dataset/dnd_2017'

    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')

    with torch.no_grad():
        for i in range(50):
            filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
            img = h5py.File(filename, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)
            ref = bb[0][i]
            boxes = np.array(info[ref]).T
            for k in range(20):
                idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
                Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()
                noisy = np.expand_dims(Inoisy_crop, axis=0)

                noisy = (torch.from_numpy(noisy).permute(0, 3, 1, 2)).to(device)

                noisy = F.pad(noisy, (p, p, p, p), mode="reflect") if p != 0 else noisy

                if args.model == 'tmbsn':
                    denoised = DN(noisy, args.h)
                else:
                    denoised = DN(noisy)

                denoised = denoised[..., p:-p, p:-p] if p != 0 else denoised

                Idenoised_crop = np.clip(denoised.permute(0, 2, 3, 1).cpu().numpy(), 0.0, 1.0)

                Idenoised_crop = np.float32(Idenoised_crop)
                save_file = os.path.join(save_dir_path, '%04d_%02d.mat'%(i+1,k+1))
                sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
                print('%s crop %d/%d' % (filename, k+1, 20))
            print('[%d/%d] %s done\n' % (i+1, 50, filename))

    out_folder = os.path.join('./benchmark', args.name)
    os.makedirs(out_folder, exist_ok=True)
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=object)
        for bb_idx in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb_idx+1)
            s = sio.loadmat(os.path.join(save_dir_path,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb_idx] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                    "israw": israw,
                    "eval_version": eval_version},
                    )

    print('Done.')


if __name__ == '__main__':
    main()
