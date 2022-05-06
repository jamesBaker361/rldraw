from string_globals import *
import torch.nn as nn
import torch.nn.parallel



class Discriminator(nn.Module):
    def __init__(self, ngpu,disc_features):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features) x 32 x 32
            nn.Conv2d(disc_features, disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*2) x 16 x 16
            nn.Conv2d(disc_features * 2, disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*4) x 8 x 8
            nn.Conv2d(disc_features * 4, disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*8) x 4 x 4
            nn.Conv2d(disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)