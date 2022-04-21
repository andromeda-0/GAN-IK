import torch.nn as nn


class cGenerator(nn.Module):
    def __init__(self, config_dim, angle_dim):
        super(cGenerator, self).__init__()
        self.config_dim = config_dim
        self.angle_dim = angle_dim

    @staticmethod
    def block(in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers


class cGenerator_0(cGenerator):
    def __init__(self, config_dim, angle_dim):
        super().__init__(config_dim, angle_dim)

        self.model = nn.Sequential(
                *self.block(config_dim, 128, normalize=False),
                *self.block(128, 256),
                *self.block(256, 512),
                *self.block(512, 1024),
                nn.Linear(1024, angle_dim),
                nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class cGenerator_1(cGenerator):
    def __init__(self, config_dim, angle_dim):
        super().__init__(config_dim, angle_dim)

        self.model = nn.Sequential(
                *self.block(self.config_dim, 256, normalize=False),
                *self.block(256, 256),
                *self.block(256, 256),
                nn.Linear(256, self.angle_dim),
                nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class cDiscriminator(nn.Module):
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
    def __init__(self, config_dim, angle_dim):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(config_dim, 32),
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
                nn.Linear(64, angle_dim)
        )

    def forward(self, x):
        return self.model(x)
