"""
PyTorch LMDB Data Loader for image denoising datasets.

Replaces the TFRecord-based data_loader.py with a pure PyTorch pipeline
that reads from LMDB databases created by make_lmdb.py.
"""

import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LMDBDataset(Dataset):
    """
    PyTorch Dataset for loading image patches from LMDB databases.

    Supports SIDD (uint8) and DND (float32) datasets with automatic
    normalization, random cropping, and augmentation.
    """

    def __init__(self, lmdb_path, patch_size, augment=True):
        """
        Args:
            lmdb_path: Path to LMDB database directory
            patch_size: Target patch size (will random crop if original is larger)
            augment: Whether to apply data augmentation (rotation + flip)
        """
        self.lmdb_path = lmdb_path
        self.patch_size = patch_size
        self.augment = augment

        # LMDB env is opened lazily per worker process. Sharing an env handle
        # across forked DataLoader workers causes segfaults, so __init__ must
        # not retain an open env.
        self.env = None

        # Read metadata via a temporary env, then close it.
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with env.begin(write=False) as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
        env.close()

        self.num_samples = meta['num_samples']
        self.original_patch_size = meta['patch_size']
        self.dtype = meta['dtype']
        self.shape = meta['shape']

        self.np_dtype = np.uint8 if self.dtype == 'uint8' else np.float32
        self.need_crop = self.patch_size < self.original_patch_size

    def _init_env(self):
        """Open the LMDB env on first access from the current (worker) process."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._init_env()
        key = f'{idx:08d}'.encode('ascii')

        with self.env.begin(write=False) as txn:
            value = txn.get(key)

        patch = np.frombuffer(value, dtype=self.np_dtype).reshape(self.shape).copy()

        # Normalize uint8 to float32 [0, 1]
        if self.dtype == 'uint8':
            patch = patch.astype(np.float32) / 255.0

        # Random crop if needed
        if self.need_crop:
            h, w, _ = patch.shape
            start_h = np.random.randint(0, h - self.patch_size + 1)
            start_w = np.random.randint(0, w - self.patch_size + 1)
            patch = patch[start_h:start_h + self.patch_size,
                         start_w:start_w + self.patch_size, :]

        # Apply augmentation (rotation + flip)
        if self.augment:
            patch = self._augment(patch)

        # Convert to torch tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).contiguous()

        return {'noisy': tensor}

    def _augment(self, patch):
        """
        Apply random augmentation matching the TF data_loader.py implementation:
        - Random rotation: 0, 90, 180, or 270 degrees
        - Random horizontal flip (50% probability)
        """
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k)

        if np.random.random() > 0.5:
            patch = np.fliplr(patch)

        return np.ascontiguousarray(patch)

    def __del__(self):
        if getattr(self, 'env', None) is not None:
            self.env.close()


def load_lmdb(lmdb_path, patch_size, batch_size, num_workers=4, shuffle=True):
    """
    Create a PyTorch DataLoader for LMDB dataset.

    Args:
        lmdb_path: Path to LMDB database directory
        patch_size: Target patch size for training
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader instance
    """
    dataset = LMDBDataset(lmdb_path, patch_size, augment=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    return dataloader


class InfiniteDataLoader:
    """
    Wrapper that makes a DataLoader iterate infinitely,
    replicating TF's dataset.repeat() behavior.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


def load_lmdb_infinite(lmdb_path, patch_size, batch_size, num_workers=4, shuffle=True):
    """
    Create an infinitely iterating DataLoader for LMDB dataset.

    This is the recommended function for training, matching the behavior
    of load_tfrecords() with dataset.repeat().

    Usage in training script:
        loader = load_lmdb_infinite(lmdb_path, patch_size, batch_size)
        for i, batch in enumerate(loader):
            noisy = batch['noisy'].to(device)  # (B, C, H, W), float32, [0,1]

    Args:
        lmdb_path: Path to LMDB database directory
        patch_size: Target patch size for training
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data

    Returns:
        InfiniteDataLoader instance
    """
    dataloader = load_lmdb(lmdb_path, patch_size, batch_size, num_workers, shuffle)
    return InfiniteDataLoader(dataloader)
