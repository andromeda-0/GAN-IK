import torch.nn as nn


class Generator_Base(nn.Module):
    def __init__(self, feature_size, latent_dim):
        super(Generator_Base, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, feature_size),
                nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator_Base(nn.Module):
    def __init__(self, feature_shape):
        super(Discriminator_Base, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(feature_shape, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)

        return validity
