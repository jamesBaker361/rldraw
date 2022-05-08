import gym
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from gym.spaces import MultiDiscrete,Tuple,Dict,Discrete,MultiBinary
from datetime import datetime
from discrete_space_fix import DiscretePrime
from ray.rllib.env.env_context import EnvContext
import os

image_size=64

class DrawingEnv(gym.Env): #most basic environemnt
    def __init__(self,config: EnvContext):
        print("Creating drawing environment")
        self.episodes_completed=0
        self.discriminator=config["discriminator"]
        self.action_space=MultiDiscrete([3,3,2])
        '''
        Horizontally, the agent can move left,right, or not at all (x dimension)
        Vertically, the agent can move up,down,right, or not at all (y dimension)

        The agent can color the current square white or black (0,1)
        '''
        self.image_size=config["image_size"]
        self.observation_space=Dict({
            "board":MultiBinary(self.image_size**2),
            "x":DiscretePrime(self.image_size),
            "y":DiscretePrime(self.image_size)
        })
        """
        The first part of the MultiDiscrete obervation space represents the coloration of the square
        The second,third part represents the (x,y) location of the agent; x is first coordinate
        """
        self.horizon=config["horizon"] #how many steps until we give up
        self.threshold =config["threshold"] #what score before we say we've done enough
        #self.state=[[1.0 for x in range(self.image_size**2)],random.randrange(0,self.image_size),random.randrange(0,self.image_size)]
        self.board=np.array([[1 for y in range(self.image_size)] for x in range(self.image_size)])
        self.x=np.random.randint(0,self.image_size)
        self.y=np.random.randint(0,self.image_size)
        self.history=[]
        self.step_count=0
        self.draw=config["draw"]

        if self.draw:
            timestamp=datetime.now()
            timestring=timestamp.strftime("%m-%d-%H:%M:%S")
            self.image_dir=config["image_dir"]+"{}_{}/".format(timestring,os.getpid())
            if not os.path.exists(self.image_dir):
                os.makedirs(self.image_dir)

    def render(self):
        plt.imshow(self.board)

    def _get_state(self):
        return {"board":self.board.flatten(),"x":self.x,"y":self.y}

    def reset(self):
        self.episodes_completed+=1
        self.history=[]
        self.step_count=0
        self.board=np.array([[0 for y in range(self.image_size)] for x in range(self.image_size)])
        self.x=np.random.randint(0,self.image_size)
        self.y=np.random.randint(0,self.image_size)
        return self._get_state()


    def step(self,action):
        #assuming action= [vertical,horizontal,color]
        self.history.append(self.board.copy())
        self.step_count+=1
        [horizontal,vertical,color]=action
        self.board[self.x][self.y]=color
        if horizontal ==2:
            self.x+=1
        elif horizontal==1:
            self.x-=1
        self.x=self.x%self.image_size

        if vertical ==2:
            self.y+=1
        elif vertical==1:
            self.y-=1
        self.y=self.y%self.image_size

        tensor=torch.from_numpy(np.expand_dims(self.board.astype(np.float32),axis=(0,1)))

        output=self.discriminator(tensor)

        reward=output.tolist()[0][0][0][0]

        done=False
        if reward>=self.threshold or self.step_count >= self.horizon:
            done=True
            if self.draw:
                img_path=self.image_dir+"board_{}.jpg".format(self.episodes_completed)
                plt.imsave(img_path,self.board *255,cmap="gray")

        return self._get_state(),reward,done,{}



