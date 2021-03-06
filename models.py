import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, i_dim, o_dim):
        super(Generator, self).__init__()
        self.i_dim = i_dim
        self.o_dim = o_dim

    @staticmethod
    def block(in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers


class Generator_0(Generator):
    def __init__(self, i_dim, o_dim):
        super().__init__(i_dim, o_dim)

        self.model = nn.Sequential(
                *self.block(i_dim, 128, normalize=False),
                *self.block(128, 256),
                *self.block(256, 512),
                *self.block(512, 1024),
                nn.Linear(1024, o_dim),
        )

    def forward(self, z):
        return self.model(z)


class Generator_1(Generator):
    def __init__(self, i_dim, o_dim):
        super().__init__(i_dim, o_dim)

        self.model = nn.Sequential(
                *self.block(self.i_dim, 256, normalize=False),
                *self.block(256, 256),
                *self.block(256, 256),
                nn.Linear(256, self.o_dim),
        )

    def forward(self, z):
        return self.model(z)


class Generator_2(Generator):
    def __init__(self, i_dim, o_dim):
        super().__init__(i_dim, o_dim)

        self.model = nn.Sequential(
                nn.Linear(self.i_dim, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, self.o_dim)
        )

    def forward(self, z):
        return self.model(z)


class Generator_3(Generator):
    def __init__(self, i_dim, o_dim):
        super().__init__(i_dim, o_dim)

        self.model = nn.Sequential(
                nn.Linear(self.i_dim, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, self.o_dim)
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, io_size):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(io_size, 256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)

        return validity


class Dense(nn.Module):
    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(i_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, o_dim)
        )

    def forward(self, x):
        return self.model(x)
