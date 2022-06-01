#from stable_baselines3.common.policies import MlpPolicy
from environments import *
from stable_baselines3 import A2C,PPO
import torch
import numpy as np
from string_globals import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--image_size",type=int,default=32,help="length of image")
parser.add_argument("--trainer",type=str,default="a2c",help="type of training algo for RL (a2c,ppo)")
parser.add_argument("--environment",type=str,default="DrawingEnv",help="environment to train agent in")
parser.add_argument("--horizon",type=int,default=500,help="how many steps per episode")
parser.add_argument("--threshold",type=float,default=0.75,help="threshold of discriminator confidence before we end the episode")
parser.add_argument("--draw",type=bool,default=True,help="whether to draw the environment at the end of each episode")
parser.add_argument("--timesteps",type=int,default=500000,help="how many timestepsin total")
parser.add_argument("--name",type=str,default="name of the folder to save images in")
parser.add_argument("--char",type=int,default=10,help="which char to use for dataset; 10 = all")
parser.add_argument("--thickness",type=int,default=1,help="thickness of brush for thickenv")
parser.add_argument("--lower_threshold",type=float,default=0.1)
parser.add_argument("--latent_dim",type=int, default=16,help="latent dim for encoded")


args = parser.parse_args()

def yesman(x):
    return torch.tensor([[[[np.random.random()]]]])

disc=torch.load(checkpoint_dir+"disc_{}_{}.pt".format(args.image_size,args.char))

env_config={
    "discriminator":disc,
    "image_size":args.image_size,
    "threshold":args.threshold,
    "horizon":args.horizon,
    "image_dir":gen_imgs_dir+"/{}/".format(args.name),
    "draw":True,
    "lower_threshold": args.lower_threshold,
    "episodes":args.timesteps // args.horizon,
    "distortion":0.25,
    "final_distortion":0.75
}

if args.environment=="HintDrawingEnv":
    from data_loading import *
    train_loader,test_loader=get_data_loaders_specific_char(mnist_dir,args.image_size,8,args.char,channels=1)
    env_config["data_loader"]=train_loader

if args.environment=="DreamEnv":
    env_config["vae"]=torch.load(checkpoint_dir+"vae_{}_{}_{}.pt".format(args.image_size,args.char,args.latent_dim))

trainer_dict={
    "a2c":A2C,
    "ppo":PPO
}

environment_dict={
    "DrawingEnv":DrawingEnv,
    "ThickDrawingEnv":ThickDrawingEnv,
    "HintDrawingEnv":HintDrawingEnv,
    "DreamEnv":DreamEnv
}

env=environment_dict[args.environment](env_config)

model = trainer_dict[args.trainer]("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=args.timesteps)