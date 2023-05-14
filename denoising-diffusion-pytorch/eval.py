import numpy as np
import torch
from torch.optim import Adam
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import cv2
import argparse


"""
pip install opencv-python
pip install accelerate
pip install importlib_metadata
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="/var/datasets/real_3_channels")
    parser.add_argument('--saving_path', type=str, default="../checkpoints/real_3_channels")
    parser.add_argument('--sample_path', type=str, default="../checkpoints/samples")
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--channel_num', type=int, default=3)

    parser.add_argument('--epoch', type=int, default=700000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=8e-5)


    args = parser.parse_args()
    return args



args = parse_args()
channel, img_size, bs = args.channel_num, args.img_size, args.bs



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=channel,
)


model.to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = args.img_size,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

diffusion.to(device)


trainer = Trainer(
    diffusion,
    args.dataset_path,
    train_batch_size = args.bs,
    train_lr = args.lr,
    train_num_steps = args.epoch,            # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    is_6_channel = args.channel_num == 6,
    results_folder = args.saving_path,
    img_size=args.img_size,
    save_and_sample_every = 1000,
    num_samples = 1,
)

trainer.eval(700, pth=args.sample_path)