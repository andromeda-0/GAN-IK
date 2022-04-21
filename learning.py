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


class Learning:
    def __init__(self, args):
        self.args = args
        run_id = str(vars(args)).encode('utf-8')
        self.config_string = hashlib.md5(run_id).hexdigest()
        with open('configs/%s.json' % self.config_string, 'w') as f:
            json.dump(vars(args), f)
        self.device = torch.device('cuda:%d' % args.gpu_id)
        self.dataset = KinematicsSet(args.data_path)
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=args.batch_size,
                shuffle=True,
        )
        self.writer = SummaryWriter('./runs/%s' % self.config_string)
        self.mean, self.std = self.dataset.mu_sigma()


class GAN(Learning):
    def __init__(self, args):
        super(GAN, self).__init__(args)

        self.adversarial_loss = torch.nn.BCELoss()
        self.generator = eval(args.generator + '(self.dataset.i_size, self.dataset.o_size)')
        self.discriminator = eval(
            args.discriminator + '(self.dataset.i_size + self.dataset.o_size)')

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.adversarial_loss.to(self.device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr_g,
                                            betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_d,
                                            betas=(args.b1, args.b2))

    def __call__(self):
        for epoch in tqdm(range(self.args.n_epochs)):
            for i, (angles, configurations) in enumerate(self.dataloader):
                I = angles.to(device=self.device)
                O = configurations.to(device=self.device)
                valid = torch.ones((I.shape[0], 1), device=self.device)
                fake = torch.zeros((I.shape[0], 1), device=self.device)

                self.optimizer_G.zero_grad()

                synthetic_i = torch.zeros((angles.shape[0], self.dataset.i_size))
                for channel in range(self.dataset.i_size):
                    synthetic_i[channel, :] = torch.normal(self.mean, self.std)
                synthetic_i = synthetic_i.to(device=self.device)

                # Generate a batch of images
                synthetic_o = self.generator(synthetic_i)
                synthetic_i_o = torch.cat([synthetic_i, synthetic_o], dim=1)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(synthetic_i_o), valid)

                g_loss.backward()
                self.optimizer_G.step()

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_i_o = torch.cat([I, O], dim=1)
                real_loss = self.adversarial_loss(self.discriminator(real_i_o), valid)
                fake_loss = self.adversarial_loss(self.discriminator(synthetic_i_o.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                self.writer.add_scalar('Discriminator Loss', d_loss.item(), epoch)
                self.writer.add_scalar('Generator Loss', g_loss.item(), epoch)


class cGAN(Learning):
    def __init__(self, args):
        super(cGAN, self).__init__(args)
