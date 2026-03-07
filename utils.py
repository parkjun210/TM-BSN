import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

import math
import glob
from shutil import copy
import cv2
import base64

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2ycbcr


def make_indexed_dir(main_name, sub_name, root_dir='./output'):

    idx = 1

    base_path = os.path.join(root_dir, main_name)

    candidate_path = os.path.join(base_path, f"{sub_name}_{idx}")

    while os.path.exists(candidate_path):
        idx += 1
        candidate_path = os.path.join(base_path, f"{sub_name}_{idx}")

    os.makedirs(candidate_path, exist_ok=True)

    os.makedirs(os.path.join(candidate_path, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(candidate_path, 'val_images'), exist_ok=True)
    os.makedirs(os.path.join(candidate_path, 'validation_samples'), exist_ok=True)

    return candidate_path

def cpy_code(checkpoint_dir):
    files = glob.glob('./*.py')
    os.makedirs(checkpoint_dir, exist_ok=True)
    for file in files:
        copy(file, os.path.join(checkpoint_dir, file))

def set_random_seed(seed):
    print(f'seed = {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_checkpoint(args, model, optimizer, scheduler, device):
    assert args.resume, "args.resume must be a valid checkpoint path"
    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        print(f"[Warning!!!] missing_keys={missing}, unexpected_keys={unexpected}")

    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])

    start_iter = int(ckpt.get('iter', -1)) + 1
    print(f"Resuming training from iteration {start_iter}")

    model.train()
    return start_iter

class CosineDecayScheduler:
    def __init__(self, optimizer, initial_lr, fixed_steps, decay_steps, alpha):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.fixed_steps = fixed_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.current_step = 0

    def step(self):
        if self.current_step < self.fixed_steps:
            lr = self.initial_lr
        else:
            global_step = self.current_step - self.fixed_steps

            if global_step >= self.decay_steps:
                lr = self.alpha
            else:
                cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / self.decay_steps))
                lr = (self.initial_lr - self.alpha) * cosine_decay + self.alpha

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def state_dict(self):
        return {
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']

def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

def calculate_psnr_255(target, ref, data_range=255.0):
    img1 = np.round(np.clip(target * 255.0, 0, 255)).astype(np.float32)
    img2 = np.round(np.clip(ref    * 255.0, 0, 255)).astype(np.float32)

    diff = img1 - img2
    mse = np.mean(np.square(diff))
    if mse == 0:
        return float('inf')
    psnr = 10.0 * np.log10(data_range**2 / mse)
    return psnr

def ssim_y_channel(out, gt):
    out_uint8 = np.round(out * 255.0).clip(0, 255).astype(np.uint8)
    gt_uint8  = np.round(gt  * 255.0).clip(0, 255).astype(np.uint8)

    out_y = rgb2ycbcr(out_uint8)[:, :, 0]
    gt_y  = rgb2ycbcr(gt_uint8)[:, :, 0]

    val = ssim(
        out_y, gt_y,
        data_range=255,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        K1=0.01, K2=0.03
    )
    return val

def eval_bsn_outs(noisy, gt, model, h_set, save_dir, count, pad=0):
    """forward_outs를 이용해 한 번의 backbone pass로 여러 h에 대한 결과를 평가"""
    if pad != 0:
        noisy = F.pad(noisy, (pad, pad, pad, pad), mode="reflect")

    outs = model.forward_outs(noisy, h_set)  # list of [B,C,H,W]

    results = {}
    for h, out in zip(h_set, outs):
        if pad != 0:
            out = out[..., pad:-pad, pad:-pad]
        out_np = np.clip(out.detach().permute(0, 2, 3, 1).cpu().numpy(), 0.0, 1.0)
        ssim_v = ssim_y_channel(out_np[0], gt[0])
        psnr_v = calculate_psnr_255(out_np, gt)
        name = f'h{h}'
        cv2.imwrite(os.path.join(save_dir, f'{count:04d}_{name}_{psnr_v:.2f}.png'),
                    cv2.cvtColor(np.round(out_np[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        results[h] = (psnr_v, ssim_v)

    return results  # dict: {h: (psnr, ssim)}

def eval_nbd_img(noisy, gt, model, save_dir, count, name, pad=0):

    if pad != 0:
        noisy = F.pad(noisy, (pad, pad, pad, pad), mode="reflect")

    out = model(noisy)
    if pad != 0:
        out = out[..., pad:-pad, pad:-pad]
    out = np.clip(out.permute(0, 2, 3, 1).cpu().numpy(), 0.0, 1.0)
    ssim_v = ssim_y_channel(out[0], gt[0])
    psnr_v = calculate_psnr_255(out, gt)

    return psnr_v, ssim_v


class Recharger:
    def __init__(self, percentage=1.0):
        self.percentage = percentage

    def generate_subset_mask(self, y):
        true_indices = torch.nonzero(y, as_tuple=False)
        num_true = true_indices.size(0)
        if num_true == 0:
            return torch.zeros_like(y, dtype=torch.bool)
        num_to_keep = int(num_true * self.percentage)
        shuffled_indices = torch.randperm(num_true, device=y.device)
        keep_indices = true_indices[shuffled_indices[:num_to_keep]]
        subset_mask = torch.zeros_like(y, dtype=torch.bool)
        subset_mask[keep_indices[:, 0], keep_indices[:, 1], keep_indices[:, 2]] = True
        return subset_mask

    def apply(self, y, T_elements, sampler=None, generator=None):
        if sampler is not None:
            y_src = sampler(y, generator=generator)
        else:
            y_src = y

        T_cat = torch.stack(T_elements, dim=0)  # (distill_no, B, C, H, W)
        distill_no, B, C, H, W = T_cat.shape
        device = y.device

        recharger = torch.randint(0, distill_no, (B, H, W), device=device, generator=generator)

        y_src = y_src.permute(1, 0, 2, 3)       # [C, B, H, W]
        T_cat = T_cat.permute(0, 2, 1, 3, 4)    # [distill_no, C, B, H, W]

        for no in range(distill_no):
            recharging_mask = (recharger == no)          # [B,H,W] (bool)
            recharging_mask = self.generate_subset_mask(recharging_mask)
            T_cat[no, :, recharging_mask] = y_src[:, recharging_mask]

        T_RD_cat = T_cat.permute(0, 2, 1, 3, 4)  # [distill_no, B, C, H, W]
        T_RD_elements = [t.squeeze(0) for t in torch.chunk(T_RD_cat, chunks=distill_no, dim=0)]
        return T_RD_elements
