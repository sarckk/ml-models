"""
    PyTorch implementation of Deep Convolutional GAN (Radford, Alec, Luke Metz, and Soumith Chintala, 2015)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        # input is of shape N x img_channels x 64 x 64
        self.disc = nn.Sequential(                              
                nn.Conv2d(img_channels, 128, 4, 2, 1),         # 128 x 32 x 32 , kernel_size=3 also works here
                nn.LeakyReLU(0.2),                          
                self.conv_block(128, 256, 4, 2, 1),            # 256 x 16 x 16
                self.conv_block(256, 512, 4, 2, 1),            # 512 x 8 x 8
                self.conv_block(512, 1024, 4, 2, 1),           # 1024 x 4 x 4
                nn.Conv2d(1024, 1, 4, 2, 0),                   # 1 x 1 x 1
                nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride, 
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels):
        super().__init__()
        # input is of shape N x z_dim x 1 x 1 
        # Formula to calculate output height and width from ConvTranspose2d:
        # out_height = strides[1] * (in_height - 1) + kernel_size[0] - 2 * padding_height
        # out_width  = strides[2] * (in_width - 1) + kernel_size[1] - 2 * padding_width
        self.gen = nn.Sequential(
                self.deconv_block(z_dim, 1024, 4, 1, 0),
                self.deconv_block(1024, 512, 4, 2, 1),
                self.deconv_block(512, 256, 4, 2, 1),
                self.deconv_block(256, 128, 4, 2, 1),
                nn.ConvTranspose2d(128, img_channels, 4, 2, 1),
                nn.Tanh()
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride, 
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        x = x.reshape(-1, x.shape[1], 1, 1)
        return self.gen(x)


def init_weights(module):
    if module.type in (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d):
        nn.init.normal_(module.weight.data, 0, 0.02)
    elif module.type == nn.BatchNorm2d:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


if __name__ == '__main__':
    # train DCGAN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_dim = 100
    n_channels = 3
    lr = 0.0002
    batch_size = 128
    betas = (0.5, 0.999)
    n_epochs = 50
    img_size = 64

    disc = Discriminator(n_channels).to(device)
    gen = Generator(z_dim, n_channels).to(device)
    
    # initialize weights with mean = 0.0 and std = 0.02
    disc.apply(init_weights)

    fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)

    transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), 
                transforms.Normalize([0.5] * n_channels, [0.5] * n_channels) # normalize pixels channel-wise to [-1,1] since we use tanh
            ] 
    )

    script_dir = os.path.dirname(__file__) # directory that this script is located at
    root = os.path.join(script_dir, '../data', 'faces') # assumes deep-learning/data/faces direcotry contains data
    
    ds = datasets.ImageFolder(root, transform = transforms)
    
    # Uncomment below 2 lines to train DCGAN on CelebA dataset
    # root =  os.path.join('..', 'dataset')
    # ds = datasets.CelebA(root, split='train', transform=transforms, download=True)

    dl = DataLoader(ds, shuffle=True, batch_size=batch_size)

    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=betas)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)

    criterion = nn.BCELoss()

    writer_fake = SummaryWriter(os.path.join(script_dir, '../runs/DCGAN/fake'))
    writer_real = SummaryWriter(os.path.join(script_dir, '../runs/DCGAN/real'))

    step = 0

    for epoch in range(n_epochs):
      for batch_idx, (real, _) in enumerate(dl):
        real = real.to(device)

        # train discriminator
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise) # batch_size x img_channels x 64 x 64
        # Combining real and fake data will make BatchNorm not work (see https://github.com/soumith/ganhacks/issues/52)
        # fake_and_real = torch.cat([real, fake.detach()]) 

        disc_pred_real = disc(real).view(-1) # batch_size
        loss_real = criterion(disc_pred_real, torch.ones_like(disc_pred_real))

        disc_pred_fake = disc(fake.detach()).view(-1) # batch_size
        loss_fake = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))

        loss_disc = (loss_real + loss_fake) / 2

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # train generator
        out = disc(fake).view(-1)
        loss_gen = criterion(out, torch.ones_like(out))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
          print(
              f"Epoch {epoch} / {n_epochs} \n",
              f"discriminator loss: {loss_disc:.4f}\t generator loss: {loss_gen:.4f}"
          )

          with torch.no_grad():
            fake = gen(fixed_noise)

            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

            writer_fake.add_image('Fake', img_grid_fake, global_step = step)
            writer_real.add_image('Real', img_grid_real, global_step = step)

            step += 1


