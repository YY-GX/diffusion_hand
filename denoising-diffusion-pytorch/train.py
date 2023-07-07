import numpy as np
import torch
from torch.optim import Adam
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="./CMU_6_channel_1000_imgs")
    parser.add_argument('--saving_path', type=str, default="./checkpoints")
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--channel_num', type=int, default=6)

    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=8e-5)


    args = parser.parse_args()
    return args


def channel6_to_imgs(path, size=128):
    ls = []
    for img in os.listdir(path):
        path_img = os.path.join(path, img)
        img_np = np.load(path_img)
        img_np = np.stack([cv2.resize(img_np[:, :, :3], dsize=(size, size), interpolation=cv2.INTER_CUBIC),
                           cv2.resize(img_np[:, :, 3:6], dsize=(size, size), interpolation=cv2.INTER_CUBIC)], 2)
        ls.append(img_np)
    return np.array(ls)


args = parse_args()
channel, img_size, bs = args.channel_num, args.img_size, args.bs


images = channel6_to_imgs(args.dataset_path, args.img_size)
images = images.reshape([-1, channel, img_size, img_size])
print("Dataset shape: ", images.shape)
images_torch = torch.from_numpy(images)
images_torch = images_torch / 255


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


model = Unet(
  dim=16,
  dim_mults=(1, 2, 4, 8, 16, 32),
  channels=channel,
)
model.to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=img_size,
    timesteps=100,   # number of steps
    loss_type='l1'    # L1 or L2
)
diffusion.to(device)

train_lr = args.lr
adam_betas = (0.9, 0.99)
optimizer = Adam(diffusion.parameters(), lr=train_lr, betas=adam_betas)
images_torch = images_torch.to(device)
for i in range(args.epoch):
    optimizer.zero_grad()
    start = (i * bs) % 5000
    loss = diffusion(images_torch[start:start+bs])
    loss.backward()
    optimizer.step()
    if (i % 100 == 0):
        print(f'{i}: {loss}')
    torch.save(diffusion.state_dict(), f'./checkpoint/cmu_6_1000.pt')