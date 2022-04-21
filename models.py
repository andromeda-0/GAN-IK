import torch.nn as nn


class cGenerator(nn.Module):
    def __init__(self, i_size, o_size):
        super(cGenerator, self).__init__()
        self.i_size = i_size
        self.o_size = o_size

    @staticmethod
    def block(in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers


class cGenerator_0(cGenerator):
    def __init__(self, i_size, o_size):
        super().__init__(i_size, o_size)

        self.model = nn.Sequential(
                *self.block(i_size, 128, normalize=False),
                *self.block(128, 256),
                *self.block(256, 512),
                *self.block(512, 1024),
                nn.Linear(1024, o_size),
                nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class cGenerator_1(cGenerator):
    def __init__(self, i_size, o_size):
        super().__init__(i_size, o_size)

        self.model = nn.Sequential(
                *self.block(self.i_size, 128, normalize=False),
                *self.block(128, 256),
                *self.block(256, 128),
                *self.block(128, 32),
                nn.Linear(32, self.o_size),
                nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class cDiscriminator(nn.Module):
    def __init__(self, io_size):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(io_size, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)

        return validity
