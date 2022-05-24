from string_globals import *
import torch.nn as nn
import torch.nn.parallel

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, ngpu,disc_features,image_size=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        layers=[
            # input is (nc) x image_size x image_size
            nn.Conv2d(1, disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features) x image_size/2 x image_size/2
            nn.Conv2d(disc_features, disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*2) x image_size/4 x image_size/4
            nn.Conv2d(disc_features * 2, disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if image_size >32:
            layers+=[
            nn.Conv2d(disc_features * 4, disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 8, 1, 4, 1, 0, bias=False)]
        else:
            layers+=[nn.Conv2d(disc_features * 4, disc_features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 8, 1, 2, 1, 0, bias=False)]
        layers+=[
            nn.Sigmoid()]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)