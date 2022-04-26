import hashlib
import json
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.io import loadmat

# noinspection PyUnresolvedReferences
from models import *


class KDCSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.i = None
        self.o = None
        self.len = None

    def mean_std_i(self):
        """
        The mean and variance of the i
        :return:
        """
        mu = torch.mean(self.i, 0)
        sigma = torch.std(self.i, 0)
        return mu, sigma

    def __getitem__(self, index):
        return (self.o[index], None, self.i[index])

    def __len__(self):
        return self.len


class KinematicsSet(KDCSet):
    def __init__(self, data_path):
        super(KinematicsSet, self).__init__(data_path)
        data = np.load(self.data_path)
        o = data['angles']
        i = data['configurations_without_noise']

        self.len = o.shape[0]
        self.i_dim = i.shape[1]
        self.o_dim = o.shape[1]

        self.o = torch.tensor(o, dtype=torch.float)
        self.i = torch.tensor(i, dtype=torch.float)


class DynamicsSet(KDCSet):
    def __init__(self, data_path):
        super().__init__(data_path)
        data = loadmat(self.data_path)
        i = np.concatenate([data['thetas'][:, 10000:90000], data['thetaDotss'][:, 10000:90000],
                            data['thetaDDotss']][:, 10000:90000], axis=0).transpose()
        o = data['torques'][:, 10000:90000].transpose()

        self.len = o.shape[0]
        self.i_dim = i.shape[1]
        self.o_dim = o.shape[1]

        self.o = torch.tensor(o, dtype=torch.float)
        self.i = torch.tensor(i, dtype=torch.float)


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
        dataset: KDCSet = eval(args.dataset + '(%s)' % args.data_path)
        random_indices = np.random.permutation(np.arange(len(dataset)))
        self.train_set = Subset(dataset, random_indices[0:int(0.8 * len(dataset))])
        self.valid_set = Subset(dataset, random_indices[int(0.8 * len(dataset)):])

        self.train_loader = DataLoader(
                self.train_set,
                batch_size=args.batch_size,
                shuffle=True,
        )

        self.valid_loader = DataLoader(self.valid_set, batch_size=args.batch_size, shuffle=False)

        self.writer = SummaryWriter('runs/' + self.config_string)
        self.mean, self.std = self.train_set.dataset.mean_std_i()
        self.current_synthetic_o = None
        self.current_real_o = None

    def get_synthetic_i(self, I):
        if self.args.z_method == 'add':
            synthetic_i = I + torch.normal(mean=torch.zeros_like(I), std=self.args.std).to(
                    device=self.device)
        elif self.args.z_method == 'minibatch':
            synthetic_i = torch.normal(mean=I, std=torch.std(I)).to(device=self.device)
        elif self.args.z_method == 'batch':
            synthetic_i = torch.normal(mean=self.mean * torch.ones(I.shape),
                                       std=self.std).to(device=self.device)
        else:
            raise ValueError()

        return synthetic_i

    @staticmethod
    def metric(o_1, o_2):
        """
        As we are solving the IK problem, we need to compare the real o and synthetic o
        :return:
        """

        return torch.sqrt(functional.mse_loss(o_1, o_2)).detach().cpu().item()

    @abstractmethod
    def _train_epoch(self, epoch):
        pass

    @abstractmethod
    def _validate_epoch(self, epoch):
        pass

    def train(self):
        rmse = []
        for epoch in tqdm(range(self.args.n_epochs)):
            self._train_epoch(epoch)
            rmse.append(self.valid(epoch))
        rmse = np.array(rmse[-50:])
        self.writer.add_scalar('Valid/Steady-RMSE-Mean', rmse.mean())
        self.writer.add_scalar('Valid/Steady-RMSE-Std', rmse.std())
        print(str(self.args).replace(' ', '') + ' mean: %.2f std: %.2f' % (rmse.mean(), rmse.std()))
        self.writer.close()

    def valid(self, epoch):
        with torch.no_grad():
            self.current_real_o = torch.zeros(
                    (len(self.valid_set),) + self.valid_set.dataset.o.shape[1:],
                    device=self.device)
            self.current_synthetic_o = torch.zeros_like(self.current_real_o)
            self._validate_epoch(epoch)
            rmse_ik = self.metric(self.current_real_o, self.current_synthetic_o)
            self.writer.add_scalar('Valid/RMSE', rmse_ik, global_step=epoch)
            self.writer.flush()
        return rmse_ik


class GAN(Learning):
    def __init__(self, args):
        super().__init__(args)

        self.adversarial_loss = torch.nn.BCELoss()
        self.generator: nn.Module = eval(
                args.generator + '(self.train_set.dataset.i_dim,'
                                 ' self.train_set.dataset.o_dim)')
        self.discriminator: nn.Module = eval(
                args.discriminator + '(self.train_set.dataset.i_dim'
                                     ' + self.train_set.dataset.o_dim)')

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
        for i, (o, _, i_no_noise) in enumerate(self.train_loader):
            # during training, we do not have access to noise-free i
            I = i_no_noise.to(device=self.device)
            O = o.to(device=self.device)

            valid = torch.ones((I.shape[0], 1), device=self.device)
            fake = torch.zeros((I.shape[0], 1), device=self.device)

            self.optimizer_G.zero_grad()

            synthetic_i = self.get_synthetic_i(I)

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
        for i, (o, _, i_no_noise) in enumerate(
                self.valid_loader):
            # during validation, we do not need the noised i
            I = i_no_noise.to(device=self.device)
            real_O = o.to(device=self.device)

            predicted_o = self.generator(I)

            self.current_real_o[i * self.args.batch_size:
                                (i + 1) * self.args.batch_size, :] = real_O
            self.current_synthetic_o[i * self.args.batch_size:
                                     (i + 1) * self.args.batch_size, :] = predicted_o


class wGAN(Learning):
    def __init__(self, args):
        super().__init__(args)

        self.generator: nn.Module = eval(
                args.generator + '(self.train_set.dataset.i_dim,'
                                 ' self.train_set.dataset.o_dim)')
        self.discriminator: nn.Module = eval(
                args.discriminator + '(self.train_set.dataset.i_dim'
                                     ' + self.train_set.dataset.o_dim)')

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
        for i, (o, _, i_no_noise) in enumerate(self.train_loader):
            # during training, we do not have access to noise-free i
            I = i_no_noise.to(device=self.device)
            O = o.to(device=self.device)

            self.optimizer_D.zero_grad()

            synthetic_i = self.get_synthetic_i(I)

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
        for i, (o, _, i_no_noise) in enumerate(
                self.valid_loader):
            # during validation, we do not need the noised i
            I = i_no_noise.to(device=self.device)
            real_O = o.to(device=self.device)

            predicted_o = self.generator(I)

            self.current_real_o[i * self.args.batch_size:
                                (i + 1) * self.args.batch_size, :] = real_O
            self.current_synthetic_o[i * self.args.batch_size:
                                     (i + 1) * self.args.batch_size, :] = predicted_o


class DiscriminativeModel(Learning):
    def __init__(self, args):
        super().__init__(args)

        self.loss = torch.nn.MSELoss()
        self.model: nn.Module = eval(
                args.discriminator + '(self.train_set.dataset.i_dim,'
                                     ' self.train_set.dataset.o_dim)')

        self.model.to(self.device)
        self.loss.to(self.device)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_d,
                                          betas=(args.b1, args.b2))

    def _train_epoch(self, epoch):
        self.model.train()
        loss_mean = 0.0
        i = 0
        for i, (o, _, i_no_noise) in enumerate(self.train_loader):
            I = i_no_noise.to(device=self.device)
            O = o.to(device=self.device)

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
        for i, (o, _, i_without_noise) in enumerate(
                self.valid_loader):
            # during validation, we do not need the noised i
            I = i_without_noise.to(device=self.device)
            real_O = o.to(device=self.device)

            predicted_o = self.model(I)

            self.current_real_o[i * self.args.batch_size:
                                (i + 1) * self.args.batch_size, :] = real_O
            self.current_synthetic_o[i * self.args.batch_size:
                                     (i + 1) * self.args.batch_size, :] = predicted_o


class ssGAN(GAN):
    def __init__(self, args):
        super().__init__(args)
        self.self_supervised_loss = nn.MSELoss()

    def _train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        i = 0
        g_loss_mean = 0
        d_loss_mean = 0
        for i, (o, _, i_no_noise) in enumerate(self.train_loader):
            # during training, we do not have access to noise-free i
            I = i_no_noise.to(device=self.device)
            O = o.to(device=self.device)

            valid = torch.ones((I.shape[0], 1), device=self.device)
            fake = torch.zeros((I.shape[0], 1), device=self.device)

            self.optimizer_G.zero_grad()

            synthetic_i = self.get_synthetic_i(I)

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
            alpha_samples = 2 * torch.rand(O.shape[0], 1) - 1
            alpha_samples = alpha_samples.to(device=self.device)
            interpolates_o = (alpha_samples * O + (1 - alpha_samples) * synthetic_o.detach())
            interpolates_i = (alpha_samples * I + (1 - alpha_samples) * synthetic_i.detach())
            g_estimates = [torch.cat([interpolates_i, interpolates_o], dim=1),
                           torch.cat([I, O], dim=1),
                           torch.cat([synthetic_i.detach(), synthetic_o.detach()], dim=1)]
            g_targets = [alpha_samples, valid, fake]

            d_loss = torch.zeros(1, device=self.device)
            for estimator_i, target_i in zip(g_estimates, g_targets):
                d_loss += self.self_supervised_loss(self.discriminator(estimator_i), target_i)

            d_loss.backward()
            d_loss_mean += d_loss.item()
            self.optimizer_D.step()

        self.writer.add_scalar('Discriminator Loss', d_loss_mean / (i + 1), epoch)
        self.writer.flush()
        self.writer.add_scalar('Generator Loss', g_loss_mean / (i + 1), epoch)
        self.writer.flush()
