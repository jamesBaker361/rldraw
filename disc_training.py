#https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from string_globals import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from torch.autograd import Variable
from torchvision.utils import save_image
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from disc import *

parser = argparse.ArgumentParser()

parser.add_argument(
	"--batch_size", type=int, default=32, help="batch size."
)

parser.add_argument(
	"--image_size", type=int, default=64, help="image l,w"
)

parser.add_argument(
    "--gen_features",type=int, default=64, help="generator features"
)

parser.add_argument("--latent_dim",type=int, default=100,help="latent dim for input to generator")

parser.add_argument("--disc_features", type=int,default =64, help="discriminator features")

parser.add_argument("--draw",type=bool, default=False,help="whether to draw some images")

parser.add_argument("--ngpu",type=int, default=0, help="# of gpus to use")

parser.add_argument("--beta",type=float, default=0.5, help="beta for ADAM")

parser.add_argument("--lr",type=float,default=0.002,help="learning rate for ADAM")

parser.add_argument("--num_epochs",type=int,default=25,help="how may epocs to train for")

parser.add_argument("--pretrain_epochs",type=int,default=0,help="how many epochs to train discriminator one")

parser.add_argument("--num_batches",type=int, default=10000,help="max batches per epoch")

parser.add_argument("--save",type=bool,default=False,help="whether to save the discirimianor")

parser.add_argument("--path",type=str,default="dc_discrim.pt",help="where to save the dscirimnator")

args = parser.parse_args()
print(f"Running with following CLI options: {args}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = args.batch_size
image_size=args.image_size
# Size of feature maps in generator
gen_features = args.gen_features

# Size of feature maps in discriminator
disc_features = args.disc_features

latent_dim=args.latent_dim

ngpu=args.ngpu

lr=args.lr

beta=args.beta

num_epochs=args.num_epochs

pretrain_epochs=args.pretrain_epochs

num_batches=args.num_batches

# MNIST Dataset
transform = transforms.Compose([
    transforms.Resize(image_size)
    ,transforms.ToTensor()
    #,transforms.Normalize(mean=(0.5), std=(0.5))
    ])

train_dataset = datasets.MNIST(root=mnist_dir, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=mnist_dir, train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

limit=min(num_batches*batch_size,len(train_loader)*batch_size)


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        layers=[
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_dim, gen_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_features * 8),
            nn.ReLU(True),
            # state size. (gen_features*8) x 4 x 4
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),
            # state size. (gen_features*4) x 8 x 8
            nn.ConvTranspose2d( gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 2),
            nn.ReLU(True),
        ]
        if image_size > 32:
            layers+=[
                # state size. (gen_features*2) x 16 x 16
                nn.ConvTranspose2d( gen_features * 2, gen_features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(gen_features),
                nn.ReLU(True),
                nn.ConvTranspose2d(gen_features, 1,4,2,1,bias=False)
            ]
        else:
            layers+=[nn.ConvTranspose2d(gen_features * 2, 1,4,2,1,bias=False)]
        layers+=[nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator(ngpu,disc_features,image_size).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

def train_step(i,data,pretrain=False):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    # Format batch
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

    # Forward pass real batch through D
    output = netD(real_cpu).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()


    ## Train with all-fake batch
    # Generate batch of latent vectors
    if pretrain:
        fake=torch.randn(b_size,1,image_size,image_size,device=device) *0.3081 + 0.1307
        #fake=torch.empty((b_size,1,image_size,image_size))._normal()
    else:
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
    
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    #print(label.size())
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    if pretrain==False:
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
    else:
        D_G_z2=0.0
        errG=0.0

        # Output training stats
    if i % 10 == 0:
        if pretrain==True:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tD(x): %.4f'
                % (epoch, num_epochs, i*batch_size, limit,
                    errD.item(),D_x))
        else:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i*batch_size, limit,
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    return errD, errG

print("Pretraining loop...")
for epoch in range(pretrain_epochs):
    for i, data in enumerate(train_loader, 0):
        if i >= num_batches:
            break
        train_step(i, data,True)


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_loader, 0):
        if i >= num_batches:
            break
        errD,errG=train_step(i, data,False)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

if args.save:
    torch.save(netD, checkpoint_dir+args.path)

if args.draw:
    real_batch=next(iter(test_loader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True)))
    plt.savefig("images.jpg")
    print(real_batch[0].shape)