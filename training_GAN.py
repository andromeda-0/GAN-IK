import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models import *


class KinematicsSet(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        angles = data['angles']
        configurations = data['configurations']

        self.len = angles.shape[0]
        self.feature_size = angles.shape[1]

        self.angles = torch.tensor(angles)
        self.configurations = torch.tensor(configurations)

    def __getitem__(self, index):
        return self.angles[index], self.configurations[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument('--data_path', default='data_7dof/data_txt.npz')
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument('--gpu_id', default=0, type=int)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:%d' % args.gpu_id)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    dataset = KinematicsSet(args.data_path)

    # Initialize generator and discriminator
    generator = Generator_Base(dataset.feature_size, args.latent_dim)
    discriminator = Discriminator_Base(dataset.feature_size)

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    writer = SummaryWriter('./runs')

    for epoch in range(args.n_epochs):
        for i, (angles, configurations) in enumerate(dataloader):
            i_o = torch.concat([angles, configurations]).to(device)
            valid = torch.ones((i_o.shape[0], 1), device=device)
            fake = torch.zeros((i_o.shape[0], 1), device=device)

            optimizer_G.zero_grad()

            z = torch.normal(0, 1, size=(i_o.shape[0], args.latent_dim), device=device)

            # Generate a batch of images
            synthetic_i_o = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(synthetic_i_o), valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(i_o), valid)
            fake_loss = adversarial_loss(discriminator(synthetic_i_o.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            writer.add_scalar('Discriminator Loss', d_loss.item(), epoch)
            writer.add_scalar('Generator Loss', g_loss.item(), epoch)
