import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from models import *


class KinematicsSet(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        angles = data['angles']
        configurations = data['configurations']

        self.len = angles.shape[0]
        self.config_dim = configurations.shape[1]
        self.angle_dim = angles.shape[1]

        self.angles = torch.tensor(angles, dtype=torch.float)
        self.configurations = torch.tensor(configurations, dtype=torch.float)
        self.configurations_without_noise = torch.tensor(data['configurations_without_noise'],
                                                         dtype=torch.float)

    def __getitem__(self, index):
        return (self.angles[index], self.configurations[index],
                self.configurations_without_noise[index])

    def __len__(self):
        return self.len

    def mean_std_config(self):
        """
        The mean and variance of the configurations
        :return:
        """
        mu = torch.mean(self.configurations, 0)
        sigma = torch.std(self.configurations, 0)
        return mu, sigma


class KinematicsSubset(Dataset):
    def __init__(self, dataset: KinematicsSet, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def configurations_shape(self):
        return (self.__len__(),) + self.dataset.configurations.shape[1:]

    def angles_shape(self):
        return (self.__len__(),) + self.dataset.angles.shape[1:]

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)


class Learning(ABC):
    def __init__(self, args):
        self.args = args
        run_id = str(vars(args))
        self.config_string = hashlib.md5(run_id.encode('utf-8')).hexdigest()
        if not os.path.exists('configs/'):
            os.mkdir('configs')
        with open('configs/%s.json' % self.config_string, 'w') as f:
            json.dump(vars(args), f)
        self.device = torch.device('cuda:%d' % args.gpu_id)
        dataset = KinematicsSet(args.data_path)
        random_indices = np.random.permutation(np.arange(len(dataset)))
        self.train_set = KinematicsSubset(dataset, random_indices[0:int(0.8 * len(dataset))])
        self.valid_set = KinematicsSubset(dataset, random_indices[int(0.8 * len(dataset)):])

        self.train_loader = DataLoader(
                self.train_set,
                batch_size=args.batch_size,
                shuffle=True,
        )

        self.valid_loader = DataLoader(self.valid_set, batch_size=args.batch_size, shuffle=False)

        self.writer = SummaryWriter('runs/' + self.config_string)
        self.mean, self.std = self.train_set.dataset.mean_std_config()
        self.current_synthetic_angles = None
        self.current_real_angles = None

    @staticmethod
    def metric(angles_1, angles_2):
        """
        As we are solving the IK problem, we need to compare the real angles and synthetic angles
        :return:
        """

        return torch.sqrt(functional.mse_loss(angles_1, angles_2)).detach().cpu().item()

    @abstractmethod
    def _train_epoch(self, epoch):
        pass

    @abstractmethod
    def _validate_epoch(self, epoch):
        pass

    def train(self):
        for epoch in tqdm(range(self.args.n_epochs)):
            self._train_epoch(epoch)
            self.valid(epoch)
        self.writer.close()

    def valid(self, epoch):
        with torch.no_grad():
            self.current_real_angles = torch.zeros(self.valid_set.angles_shape(),
                                                   device=self.device)
            self.current_synthetic_angles = torch.zeros_like(self.current_real_angles)
            self._validate_epoch(epoch)
            rmse_ik = self.metric(self.current_real_angles, self.current_synthetic_angles)
            self.writer.add_scalar('Valid/RMSE', rmse_ik, global_step=epoch)
            self.writer.flush()


class GAN(Learning):
    def __init__(self, args):
        super().__init__(args)

        self.adversarial_loss = torch.nn.BCELoss()
        self.generator: nn.Module = eval(
                args.generator + '(self.train_set.dataset.config_dim,'
                                 ' self.train_set.dataset.angle_dim)')
        self.discriminator: nn.Module = eval(
                args.discriminator + '(self.train_set.dataset.config_dim'
                                     ' + self.train_set.dataset.angle_dim)')

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.adversarial_loss.to(self.device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr_g,
                                            betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_d,
                                            betas=(args.b1, args.b2))

    def _train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        i = 0
        g_loss_mean = 0
        d_loss_mean = 0
        for i, (angles, configurations, _) in enumerate(self.train_loader):
            # during training, we do not have access to noise-free configurations
            I = configurations.to(device=self.device)
            O = angles.to(device=self.device)

            valid = torch.ones((I.shape[0], 1), device=self.device)
            fake = torch.zeros((I.shape[0], 1), device=self.device)

            self.optimizer_G.zero_grad()

            synthetic_i = torch.zeros((configurations.shape[0],) +
                                      self.train_set.configurations_shape()[1:])
            for channel in range(synthetic_i.shape[0]):
                # TODO: change this to be about the minibatch
                synthetic_i[channel, :] = torch.normal(self.mean, self.std)
            # clip from -pi to pi
            synthetic_i = torch.clip(synthetic_i, -torch.pi, torch.pi)

            synthetic_i = synthetic_i.to(device=self.device)

            # Generate a batch of images
            synthetic_o = self.generator(synthetic_i)

            synthetic_i_o = torch.cat([synthetic_i, synthetic_o], dim=1)

            # Loss measures generator's ability to fool the discriminator
            g_loss = self.adversarial_loss(self.discriminator(synthetic_i_o), valid)

            g_loss.backward()
            g_loss_mean += g_loss.item()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_i_o = torch.cat([I, O], dim=1)
            real_loss = self.adversarial_loss(self.discriminator(real_i_o), valid)
            fake_loss = self.adversarial_loss(self.discriminator(synthetic_i_o.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            d_loss_mean += d_loss.item()
            self.optimizer_D.step()

        self.writer.add_scalar('Discriminator Loss', d_loss_mean / (i + 1), epoch)
        self.writer.flush()
        self.writer.add_scalar('Generator Loss', g_loss_mean / (i + 1), epoch)
        self.writer.flush()

    def _validate_epoch(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        for i, (angles, configurations, _) in enumerate(
                self.valid_loader):
            # during validation, we do not need the noised configuration
            I = configurations.to(device=self.device)
            real_O = angles.to(device=self.device)

            predicted_o = self.generator(I)

            self.current_real_angles[i * self.args.batch_size:
                                     (i + 1) * self.args.batch_size, :] = real_O
            self.current_synthetic_angles[i * self.args.batch_size:
                                          (i + 1) * self.args.batch_size, :] = predicted_o


class wGAN(Learning):
    def __init__(self, args):
        super().__init__(args)

        self.generator: nn.Module = eval(
                args.generator + '(self.train_set.dataset.config_dim,'
                                 ' self.train_set.dataset.angle_dim)')
        self.discriminator: nn.Module = eval(
                args.discriminator + '(self.train_set.dataset.config_dim'
                                     ' + self.train_set.dataset.angle_dim)')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr_g,
                                            betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_d,
                                            betas=(args.b1, args.b2))

    def _train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        g_loss_mean = 0
        d_loss_mean = 0
        i = 0
        for i, (angles, configurations, _) in enumerate(self.train_loader):
            # during training, we do not have access to noise-free configurations
            I = configurations.to(device=self.device)
            O = angles.to(device=self.device)

            self.optimizer_D.zero_grad()

            synthetic_i = torch.zeros((configurations.shape[0],) +
                                      self.train_set.configurations_shape()[1:])
            for channel in range(synthetic_i.shape[0]):
                # TODO: change this to be about the minibatch
                synthetic_i[channel, :] = torch.normal(self.mean, self.std)
            # clip from -pi to pi
            synthetic_i = torch.clip(synthetic_i, -torch.pi, torch.pi)
            synthetic_i = synthetic_i.to(device=self.device)

            # Generate a batch of images
            synthetic_o = self.generator(synthetic_i)

            synthetic_i_o = torch.cat([synthetic_i, synthetic_o], dim=1)
            real_i_o = torch.cat([I, O], dim=1)

            d_loss = -torch.mean(self.discriminator(real_i_o)) + torch.mean(
                    self.discriminator(synthetic_i_o))
            d_loss.backward()
            self.optimizer_D.step()
            d_loss_mean += d_loss.item()

            if i % self.args.n_critic == 0:
                self.optimizer_G.zero_grad()
                synthetic_o = self.generator(synthetic_i)
                synthetic_i_o = torch.cat([synthetic_i, synthetic_o], dim=1)
                g_loss = -torch.mean(self.discriminator(synthetic_i_o))
                g_loss.backward()
                self.optimizer_G.step()
                g_loss_mean += g_loss.item()

        self.writer.add_scalar('Discriminator Loss', d_loss_mean / (i + 1), epoch)
        self.writer.flush()
        self.writer.add_scalar('Generator Loss',
                               g_loss_mean / (i + 1) * self.args.n_critic, epoch)
        self.writer.flush()

    def _validate_epoch(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        for i, (angles, _, configurations_without_noise) in enumerate(
                self.valid_loader):
            # during validation, we do not need the noised configuration
            I = configurations_without_noise.to(device=self.device)
            real_O = angles.to(device=self.device)

            predicted_o = self.generator(I)

            self.current_real_angles[i * self.args.batch_size:
                                     (i + 1) * self.args.batch_size, :] = real_O
            self.current_synthetic_angles[i * self.args.batch_size:
                                          (i + 1) * self.args.batch_size, :] = predicted_o


class DiscriminativeModel(Learning):
    def __init__(self, args):
        super().__init__(args)

        self.loss = torch.nn.MSELoss()
        self.model: nn.Module = eval(
                args.discriminator + '(self.train_set.dataset.config_dim,'
                                     ' self.train_set.dataset.angle_dim)')

        self.model.to(self.device)
        self.loss.to(self.device)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_d,
                                          betas=(args.b1, args.b2))

    def _train_epoch(self, epoch):
        self.model.train()
        loss_mean = 0.0
        i = 0
        for i, (angles, configurations, _) in enumerate(self.train_loader):
            I = configurations.to(device=self.device)
            O = angles.to(device=self.device)

            self.optimizer.zero_grad()

            # Generate a batch of images
            synthetic_o = self.model(I)

            # Loss measures generator's ability to fool the discriminator
            loss = self.loss(synthetic_o, O)

            loss.backward()
            self.optimizer.step()

            loss_mean += loss.item()

        self.writer.add_scalar('Discriminator Loss', loss_mean / (i + 1), epoch)
        self.writer.flush()

    def _validate_epoch(self, epoch):
        self.model.eval()
        for i, (angles, configurations, configurations_without_noise) in enumerate(
                self.valid_loader):
            # during validation, we do not need the noised configuration
            I = configurations.to(device=self.device)
            real_O = angles.to(device=self.device)

            predicted_o = self.model(I)

            self.current_real_angles[i * self.args.batch_size:
                                     (i + 1) * self.args.batch_size, :] = real_O
            self.current_synthetic_angles[i * self.args.batch_size:
                                          (i + 1) * self.args.batch_size, :] = predicted_o
