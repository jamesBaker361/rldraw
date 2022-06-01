import torch
import torchvision
import argparse
import torch.optim as optim

from vanilla_vae import VanillaVAE

from data_loading import *
from torchsummary import summary

parser = argparse.ArgumentParser()

parser.add_argument(
	"--batch_size", type=int, default=32, help="batch size."
)

parser.add_argument(
	"--image_size", type=int, default=64, help="image l,w"
)

parser.add_argument("--latent_dim",type=int, default=16,help="latent dim for encoded")

parser.add_argument("--disc_features", type=int,default =64, help="discriminator features")

parser.add_argument("--draw",type=bool, default=False,help="whether to draw some images")

parser.add_argument("--ngpu",type=int, default=0, help="# of gpus to use")

parser.add_argument("--beta",type=float, default=0.5, help="beta for ADAM")

parser.add_argument("--lr",type=float,default=0.002,help="learning rate for ADAM")

parser.add_argument("--num_epochs",type=int,default=1,help="how may epocs to train for")

parser.add_argument("--num_batches",type=int, default=10,help="max batches per epoch")

parser.add_argument("--save",type=bool,default=False,help="whether to save the vae")

parser.add_argument("--path",type=str,default="vae.pt",help="where to save the vae")

parser.add_argument("--char",type=int,default=10,help="which char to use for dataset")

parser.add_argument("--hidden_dims",type=int,nargs="+",default=[32, 64, 128, 256, 512],help="hidden dims to encode/decode")

parser.add_argument("--model",type=str,default="vanilla",help="which vae model to use")

args,_ = parser.parse_known_args()
print(f"Running with following CLI options: {args}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = args.batch_size
image_size=args.image_size

latent_dim=args.latent_dim

ngpu=args.ngpu

lr=args.lr

beta=args.beta

num_epochs=args.num_epochs

num_batches=args.num_batches

hidden_dims=args.hidden_dims

model_dict={
    "vanilla":VanillaVAE
}

vae=model_dict[args.model](1,latent_dim,image_size,hidden_dims).to(device)

train_loader,test_loader=get_data_loaders_specific_char(mnist_dir,image_size,batch_size,args.char,channels=1)

limit=len(train_loader)*batch_size

optimizer=optim.Adam(vae.parameters(),lr=lr, betas=(beta, 0.999))

def train_step(data,i):
    vae.zero_grad()
    data=data[0].to(device)

    [recons,input,mu,log_var]=vae.forward(data)

    loss=vae.loss_function(recons,input,mu,log_var,M_N=0.1)

    loss.backward()

    optimizer.step()
    return loss

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        loss=train_step(data,i)
        if i%10==0:
            print("[{}/{}] [{}/{}] loss: {}".format(epoch,num_epochs,i*batch_size,limit,loss))

if args.save:
    torch.save(checkpoint_dir+args.path)

print("all done!")