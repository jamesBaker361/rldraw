#from stable_baselines3.common.policies import MlpPolicy
from environments import DrawingEnv
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
parser.add_argument("--threshold",type=float,default=0.99,help="threshold of discriminator confidence before we end the episode")
parser.add_argument("--draw",type=bool,default=True,help="whether to draw the environment at the end of each episode")
parser.add_argument("--timesteps",type=int,default=500000,help="how many timestepsin total")

args = parser.parse_args()

def yesman(x):
    return torch.tensor([[[[np.random.random()]]]])

disc=torch.load(checkpoint_dir+"disc{}.pt".format(args.image_size))

env_config={
    "discriminator":disc,
    "image_size":args.image_size,
    "threshold":args.threshold,
    "horizon":args.horizon,
    "image_dir":gen_imgs_dir,
    "draw":True
}

trainer_dict={
    "a2c":A2C,
    "ppo":PPO
}

environment_dict={
    "DrawingEnv":DrawingEnv
}

env=environment_dict[args.environment](env_config)

model = trainer_dict[args.trainer]("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=args.timesteps)