#https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
#typing-extensions setuptools pytorch torchvision tensorboard pandas scipy

scratch_dir = "/../../../../scratch/jlb638/"
mnist_dir=scratch_dir+"mnist_dir/"
checkpoint_dir=scratch_dir+"rldraw/checkpoints/"

import os
for d in [mnist_dir,checkpoint_dir]:
    if not os.path.exists(d):
        os.makedirs(d)