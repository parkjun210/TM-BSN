# TM-BSN: Triangular-Masked Blind-Spot Network for Self-Supervised Real-World Image Denoising

Official PyTorch implementation of **"TM-BSN: Triangular-Masked Blind-Spot Network for Self-Supervised Real-World Image Denoising"** (CVPR 2026).


## Environment Setup

```bash
git clone https://github.com/parkjun210/TM-BSN.git
cd TM-BSN
conda create -n tmbsn python=3.11 -y
conda activate tmbsn
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python wandb tqdm scikit-image scipy h5py matplotlib pandas einops lmdb
```


## Dataset Preparation

### SIDD

1. Download the [SIDD Medium Dataset](https://www.eecs.yorku.ca/~ka101/Datasets/SIDD/) (sRGB, Medium) and extract it under `./dataset/SIDD_Medium_Srgb/`.
2. Download the SIDD validation and benchmark `.mat` files and place them directly under `./dataset/`.

### DND

Download the [DND dataset](https://noise.visinf.tu-darmstadt.de/) and place it under `./dataset/dnd_2017/`.

### Expected folder structure

```
dataset/
├── SIDD_Medium_Srgb/
│   └── Data/
│       ├── 0001_001_S6_00100_00060_3200_L/
│       │   ├── *NOISY*.PNG
│       │   └── ...
│       └── ...
├── ValidationNoisyBlocksSrgb.mat
├── ValidationGtBlocksSrgb.mat
├── BenchmarkNoisyBlocksSrgb.mat
└── dnd_2017/
    ├── images_srgb/
    │   ├── 0001.mat
    │   └── ...
    └── info.mat
```


## LMDB Creation

Convert raw images into LMDB format for efficient training:

```bash
# SIDD
python make_lmdb.py --dataset SIDD --patch_size 256 --stride 128

# DND
python make_lmdb.py --dataset DND --patch_size 256 --stride 96
```

Output LMDB databases will be saved to `./dataset/lmdb/`.


## Training TM-BSN

Train the TM-BSN teacher model:

```bash
python train_tmbsn.py \
  --lmdb ./dataset/lmdb/SIDD_srgb_p256_s128.lmdb \
  --dataset SIDD \
  --batchsize 4 \
  --patchsize 128 \
  --maxiter 500000 \
  --lr 1e-4 \
  --gpu 0 \
  --main_name tmbsn \
  --sub_name sidd
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--lmdb` | (required) | Path to LMDB directory |
| `--dataset` | `SIDD` | Dataset type (`SIDD` or `DND`) |
| `--batchsize` | `4` | Training batch size |
| `--patchsize` | `128` | Training patch size |
| `--maxiter` | `500000` | Total training iterations |
| `--val_every` | `10000` | Validation frequency |
| `--h_set` | `[1,2,3,4,5,6]` | Hole sizes for validation |
| `--resume` | `None` | Checkpoint path to resume training |

Checkpoints and logs are saved under `./output/{main_name}/{sub_name}/`.


## Knowledge Distillation

Distill TM-BSN into a lightweight student network (NBSN, 1.02M params):

```bash
python train_distill.py \
  --lmdb ./dataset/lmdb/SIDD_srgb_p256_s128.lmdb \
  --dataset SIDD \
  --teacher_ckpt ./output/tmbsn/sidd_1/ckpt/best_h2.pth \
  --h_set 2 3 4 5 6 \
  --recharge \
  --maxiter 500000 \
  --gpu 0 \
  --main_name nbsn_distill \
  --sub_name sidd
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher_ckpt` | (required) | Path to pretrained TM-BSN checkpoint |
| `--h_set` | `[2,3,4,5,6]` | Hole sizes for teacher multi-scale outputs |
| `--recharge` | `False` | Enable Recharger to mix teacher/student outputs |

---

## Validation

Validate on the SIDD validation set (1280 patches):

```bash
# TM-BSN (teacher)
python validation.py --ckpt ./ckpt/TMBSN_SIDD.pth --model tmbsn --gpu 0

# NBSN (student)
python validation.py --ckpt ./ckpt/NBSN_SIDD.pth --model nbsn --gpu 0
```

Results (PSNR / SSIM) are printed to the console and saved under `./validation/`.


## Benchmark

### SIDD Benchmark

Generate a CSV submission file for the [SIDD Benchmark](https://www.eecs.yorku.ca/~ka101/Datasets/SIDD/):

```bash
python benchmark_sidd.py \
  --ckpt ./ckpt/NBSN_SIDD.pth \
  --model nbsn \
  --name nbsn_sidd \
  --pad 16 \
  --gpu 0
```

The submission file `SubmitSrgb.csv` will be saved under `./benchmark/{name}/`.

### DND Benchmark

Generate `.mat` files for submission to the [DND Benchmark](https://noise.visinf.tu-darmstadt.de/):

```bash
python benchmark_dnd.py \
  --ckpt ./ckpt/NBSN_DND.pth \
  --model nbsn \
  --name nbsn_dnd \
  --gpu 0
```

Output `.mat` files will be saved under `./benchmark/{name}/`.


## Pretrained Checkpoints

Pretrained weights are available in the `./ckpt/` directory.

**Note on `--pad`:** All benchmark numbers below are reported with `--pad 16`. In practice, running inference with `--pad 0` yields essentially the same performance, and for TM-BSN the benchmark score can even be higher depending on the hole size. We report `pad=16` as the standard setting because it mitigates boundary effects during knowledge distillation, so the student (NBSN) is trained and evaluated under the same padding configuration.


| Checkpoint | Model | Dataset | Params | PSNR (Bench) | SSIM (Bench) |
|:----------:|:-----:|:-------:|:------:|:------------:|:------------:|
| `TMBSN_SIDD.pth` | TM-BSN (s=4) | SIDD | 1.35M | 37.71 | 0.881 |
| `NBSN_SIDD.pth` | TM-BSN (D) / NBSN | SIDD | 1.02M | 38.31 | 0.900 |
| `TMBSN_DND.pth` | TM-BSN (s=4) | DND | 1.35M | 38.54 | 0.937 |
| `NBSN_DND.pth` | TM-BSN (D) / NBSN | DND | 1.02M | 39.41 | 0.949 |

