import gym
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from gym.spaces import MultiDiscrete,Tuple,Dict,Discrete,MultiBinary,Box
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
        self.action_space=MultiDiscrete([3,3,3])
        '''
        Horizontally, the agent can move left,right, or not at all (x dimension)
        Vertically, the agent can move up,down,right, or not at all (y dimension)

        The agent can color the current square white or black (0,1) or not at all (2)
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
        self.lower_threshold=config["lower_threshold"] #scores below this count as 0
        #self.state=[[1.0 for x in range(self.image_size**2)],random.randrange(0,self.image_size),random.randrange(0,self.image_size)]
        self.board=np.array([[1 for y in range(self.image_size)] for x in range(self.image_size)])
        self.action_board=np.array([[0.5 for y in range(self.image_size)] for x in range(self.image_size)])
        self.initial_board=self.board.copy()
        self.x=0
        self.y=0
        self.history=[]
        self.step_count=0
        self.draw=config["draw"]

        if self.draw:
            self.image_dir=config["image_dir"]
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
        self.initial_board=self.board.copy()
        self.action_board=np.array([[0.5 for y in range(self.image_size)] for x in range(self.image_size)])
        self.x=0
        self.y=0
        return self._get_state()


    def step(self,action):
        #assuming action= [vertical,horizontal,color]
        #self.history.append(self.board.copy())
        self.step_count+=1
        [horizontal,vertical,color]=action
        self.board[self.x][self.y]=color
        self.action_board[self.x][self.y]=color
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

        if reward<self.lower_threshold:
            reward=0

        done=False
        if reward>=self.threshold or self.step_count >= self.horizon:
            done=True
            if self.draw:
                img_path=self.image_dir+"board_{}_reward={}.jpg".format(self.episodes_completed,reward)
                img=[]
                for x in range(self.image_size):
                    img.append([])
                    for y in range(self.image_size):
                        if self.board[x][y]==0:
                            img[x].append(0)
                        else:
                            img[x].append(255)
                self.action_board*=255
                self.initial_board*=255
                big_img=np.concatenate([self.initial_board,self.action_board,img],axis=-1)
                plt.imsave(img_path,big_img,cmap="gray")

        return self._get_state(),reward,done,{}



class ThickDrawingEnv(DrawingEnv):
    #same as DrawingEnv but with adjustable thickness of the brush
    def __init__(self,config: EnvContext):
        self.thickness=config["thickness"]
        super().__init__(config)


    def step(self,action):
        #assuming action= [vertical,horizontal,color]
        self.history.append(self.board.copy())
        self.step_count+=1
        [horizontal,vertical,color]=action
        if color!=2:
            for h in range(self.x,self.x+self.thickness):
                for v in range(self.y,self.y+self.thickness):
                    if h <self.image_size and v <self.image_size:
                        self.board[h][v]=color
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

        if reward<self.lower_threshold:
            reward=0

        done=False
        if reward>=self.threshold or self.step_count >= self.horizon:
            done=True
            if self.draw and self.episodes_completed %3==0:
                img_path=self.image_dir+"board_{}.jpg".format(self.episodes_completed)
                plt.imsave(img_path,self.board *255,cmap="gray")

        return self._get_state(),reward,done,{}


class StrokeDrawingEnv(DrawingEnv):
    #instead of doing little dots, this env uses strokes.
    def __init__():
        pass

class HintDrawingEnv(DrawingEnv):
    #this uses initital states that are progressively further and further from initial drawings

    def __init__(self,config: EnvContext):
        self.data_loader=config["data_loader"]
        self.distortion=config["distortion"] #percent of initial characters to invert
        self.final_distortion=config["final_distortion"] #perent of final characters to invert
        self.distortion_rate= (self.final_distortion-self.distortion)/config["episodes"]
        super().__init__(config)

    def reset(self):
        self.episodes_completed+=1
        self.history=[]
        self.step_count=0
        random_int=np.random.randint(1,len(self.data_loader)-2)
        for x,data in enumerate(self.data_loader):
            if x>=random_int:
                break
        self.board=data[0][0][0].numpy()
        self.distortion+=self.distortion_rate
        limit=int(self.distortion*self.image_size)
        for x in range(limit):
            for y in range(limit):
                self.board[x][y]=0
        self.initial_board=self.board.copy()
        self.action_board=np.array([[0.5 for y in range(self.image_size)] for x in range(self.image_size)])
        self.x=0
        self.y=0
        return self._get_state()


class DreamEnv(HintDrawingEnv):
    def __init__(self,config):
        super().__init__(config)
        self.latent_dim=config["latent_dim"]
        self.vae=config["vae"]
        self.action_space=Dict({
            "color":DiscretePrime(2), #0=black 1=white
            "x":DiscretePrime(self.image_size),
            "y":DiscretePrime(self.image_size)
        })
        self.observation_space=Box([-5 for _ in range(2*self.latent_dim)],[5 for _ in range(2*self.latent_dim)])


    def step(self,action):
        [color,x,y]=action
        self.board[x][y]=color
        self.action_board[x][y]=color
        tensor=torch.from_numpy(np.expand_dims(self.board.astype(np.float32),axis=(0,1)))

        output=self.discriminator(tensor)

        reward=output.tolist()[0][0][0][0]

        if reward<self.lower_threshold:
            reward=0

        done=False
        if reward>=self.threshold or self.step_count >= self.horizon:
            done=True
            if self.draw:
                img_path=self.image_dir+"board_{}_reward={}.jpg".format(self.episodes_completed,reward)
                img=[]
                for x in range(self.image_size):
                    img.append([])
                    for y in range(self.image_size):
                        if self.board[x][y]==0:
                            img[x].append(0)
                        else:
                            img[x].append(255)
                self.action_board*=255
                self.initial_board*=255
                big_img=np.concatenate([self.initial_board,self.action_board,img],axis=-1)
                plt.imsave(img_path,big_img,cmap="gray")

        return self._get_state(),reward,done,{}



    def _get_state(self):
        np_board=np.expand_dims(self.board,axis=(0,1))
        tensor=torch.from_numpy(np_board)
        [mu,log_var]=self.vae.encode(tensor)
        obs=np.concatenate([mu[0].detach().numpy(),log_var[0].detach().numpy()])
        return {"observation_space":obs}