import argparse
import numpy as np
import os
import glob
from skimage import io
import random
from tqdm import tqdm
import h5py
import lmdb
import pickle

# Commit interval for periodic transaction commits to spread I/O load
COMMIT_INTERVAL = 10000


def make_lmdb_SIDD(train_dir, patch_size, stride, offset=10):
    os.makedirs('./dataset/lmdb', exist_ok=True)
    output_path = f'./dataset/lmdb/SIDD_srgb_p{patch_size}_s{stride}.lmdb'

    noisy_files = sorted(glob.glob(os.path.join(train_dir, '*/*NOISY*.PNG')))

    n_files = len(noisy_files)
    idx = list(range(n_files))
    random.shuffle(idx)

    # First pass: count total patches to estimate map_size
    print("Counting total patches...")
    total_patches = 0
    for k in tqdm(range(n_files)):
        noisy_img = io.imread(noisy_files[k])
        h, w, _ = noisy_img.shape
        n_h = len(range(offset, h - patch_size - offset + 1, stride))
        n_w = len(range(offset, w - patch_size - offset + 1, stride))
        total_patches += n_h * n_w

    # Estimate map_size: patch_size * patch_size * 3 channels * 1 byte (uint8) + overhead
    patch_bytes = patch_size * patch_size * 3 * 1
    map_size = int(total_patches * (patch_bytes + 100) * 1.2)  # 20% extra

    print(f"Estimated total patches: {total_patches}")
    print(f"Map size: {map_size / (1024**3):.2f} GB")

    env = lmdb.open(output_path, map_size=map_size)
    patch_count = 0
    txn = env.begin(write=True)

    for k in tqdm(idx):
        noisy_set = []

        noisy_img = io.imread(noisy_files[k])
        h, w, c = noisy_img.shape

        for i in range(offset, h - patch_size - offset + 1, stride):
            for j in range(offset, w - patch_size - offset + 1, stride):
                patch = noisy_img[i:i + patch_size, j:j + patch_size, :]
                noisy_set.append(patch)

        random.shuffle(noisy_set)

        num_data = len(noisy_set)

        for i in range(num_data):
            key = f'{patch_count:08d}'.encode('ascii')
            value = noisy_set[i].astype(np.uint8).tobytes()
            txn.put(key, value)
            patch_count += 1

            if patch_count % COMMIT_INTERVAL == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f'Committed {patch_count} patches...')

        print('processing {} / {} : {} patches'.format(k, n_files, num_data))

    txn.commit()

    # Store metadata
    with env.begin(write=True) as txn:
        meta = {
            'num_samples': patch_count,
            'patch_size': patch_size,
            'dtype': 'uint8',
            'shape': (patch_size, patch_size, 3)
        }
        txn.put(b'__meta__', pickle.dumps(meta))

    print(f'# of total patches: {patch_count}')
    env.close()
    print(f'Saved to {output_path}')


def make_lmdb_DND(train_dir, patch_size, stride, offset=10):
    os.makedirs('./dataset/lmdb', exist_ok=True)
    output_path = f'./dataset/lmdb/DND_p{patch_size}_s{stride}.lmdb'

    mat_files = glob.glob(os.path.join(train_dir, '*.mat'))

    n_files = len(mat_files)
    idx = list(range(n_files))
    random.shuffle(idx)

    # First pass: count total patches to estimate map_size
    print("Counting total patches...")
    total_patches = 0
    for k in tqdm(range(n_files)):
        with h5py.File(mat_files[k], 'r') as f:
            img = np.array(f['InoisySRGB'], dtype=np.float32)
            img = np.transpose(img, [2, 1, 0])
        h, w, _ = img.shape
        n_h = len(range(offset, h - patch_size - offset + 1, stride))
        n_w = len(range(offset, w - patch_size - offset + 1, stride))
        total_patches += n_h * n_w

    # Estimate map_size: patch_size * patch_size * 3 channels * 4 bytes (float32) + overhead
    patch_bytes = patch_size * patch_size * 3 * 4
    map_size = int(total_patches * (patch_bytes + 100) * 1.2)

    print(f"Estimated total patches: {total_patches}")
    print(f"Map size: {map_size / (1024**3):.2f} GB")

    env = lmdb.open(output_path, map_size=map_size)
    patch_count = 0
    txn = env.begin(write=True)

    for k in tqdm(idx):
        img_set = []
        with h5py.File(mat_files[idx[k]], 'r') as f:
            img = np.array(f['InoisySRGB'], dtype=np.float32)
            img = np.transpose(img, [2, 1, 0])

        h, w, c = img.shape
        assert c == 3

        for i in range(offset, h - patch_size - offset + 1, stride):
            for j in range(offset, w - patch_size - offset + 1, stride):
                img_set.append(img[i:i + patch_size, j:j + patch_size, :])

        random.shuffle(img_set)

        num_data = len(img_set)

        for i in range(num_data):
            key = f'{patch_count:08d}'.encode('ascii')
            value = img_set[i].astype(np.float32).tobytes()
            txn.put(key, value)
            patch_count += 1

            if patch_count % COMMIT_INTERVAL == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f'Committed {patch_count} patches...')

        print('processing {} / {} : {} patches'.format(k, n_files, num_data))

    txn.commit()

    # Store metadata
    with env.begin(write=True) as txn:
        meta = {
            'num_samples': patch_count,
            'patch_size': patch_size,
            'dtype': 'float32',
            'shape': (patch_size, patch_size, 3)
        }
        txn.put(b'__meta__', pickle.dumps(meta))

    print(f'# of total patches: {patch_count}')
    env.close()
    print(f'Saved to {output_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LMDB for training")
    parser.add_argument("--dataset", type=str, required=True, choices=['SIDD', 'DND'], help='dataset type')
    parser.add_argument("--patch_size", type=int, default=256, help='patch size')
    parser.add_argument("--stride", type=int, default=96, help='stride')
    args = parser.parse_args()

    if args.dataset == 'SIDD':
        make_lmdb_SIDD(train_dir='./dataset/SIDD_Medium_Srgb/Data', patch_size=args.patch_size, stride=args.stride)
    else:
        make_lmdb_DND(train_dir='./dataset/dnd_2017/images_srgb', patch_size=args.patch_size, stride=args.stride)
    print('Generating completed!')
