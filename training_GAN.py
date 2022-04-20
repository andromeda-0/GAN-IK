import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hashlib
import json

# noinspection PyUnresolvedReferences
from models import *


class KinematicsSet(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        angles = data['angles']
        configurations = data['configurations']

        self.len = angles.shape[0]
        self.i_size = angles.shape[1]
        self.o_size = configurations.shape[1]

        self.angles = torch.tensor(angles, dtype=torch.float)
        self.configurations = torch.tensor(configurations, dtype=torch.float)

    def __getitem__(self, index):
        return self.angles[index], self.configurations[index]

    def __len__(self):
        return self.len

    def mu_sigma(self):
        """
        The mean and variance of the angles
        :return:
        """
        mu = torch.mean(self.angles, 0)
        sigma = torch.std(self.angles, 0)
        return mu, sigma


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument('--data_path', default='data_7dof/data_txt.npz')
    parser.add_argument('--generator', default='Generator_0')
    parser.add_argument('--discriminator', default='Discriminator')
    parser.add_argument('--gpu_id', default=0, type=int)
    args = parser.parse_args()

    run_id = str(vars(args)).encode('utf-8')
    config_string = hashlib.md5(run_id).hexdigest()
    with open('configs/%s.json' % config_string, 'w') as f:
        json.dump(vars(args), f)

    device = torch.device('cuda:%d' % args.gpu_id)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    dataset = KinematicsSet(args.data_path)

    # Initialize generator and discriminator
    generator = eval(args.generator + '(dataset.i_size, dataset.o_size)')
    discriminator = eval(args.discriminator + '(dataset.i_size + dataset.o_size)')

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d,
                                   betas=(args.b1, args.b2))

    writer = SummaryWriter('./runs/%s' % config_string)

    mean, std = dataset.mu_sigma()

    for epoch in tqdm(range(args.n_epochs)):
        for i, (angles, configurations) in enumerate(dataloader):
            I = angles.to(device=device)
            O = configurations.to(device=device)
            valid = torch.ones((I.shape[0], 1), device=device)
            fake = torch.zeros((I.shape[0], 1), device=device)

            optimizer_G.zero_grad()

            synthetic_i = torch.zeros((angles.shape[0], dataset.i_size))
            for channel in range(dataset.i_size):
                synthetic_i[channel, :] = torch.normal(mean, std)
            synthetic_i = synthetic_i.to(device=device)

            # Generate a batch of images
            synthetic_o = generator(synthetic_i)
            synthetic_i_o = torch.cat([synthetic_i, synthetic_o], dim=1)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(synthetic_i_o), valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_i_o = torch.cat([I, O], dim=1)
            real_loss = adversarial_loss(discriminator(real_i_o), valid)
            fake_loss = adversarial_loss(discriminator(synthetic_i_o.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            writer.add_scalar('Discriminator Loss', d_loss.item(), epoch)
            writer.add_scalar('Generator Loss', g_loss.item(), epoch)
