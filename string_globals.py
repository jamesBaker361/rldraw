#https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
#typing-extensions setuptools pytorch torchvision tensorboard pandas scipy

scratch_dir = "/../../../../scratch/jlb638/"
mnist_dir=scratch_dir+"mnist_dir/"
checkpoint_dir=scratch_dir+"rldraw/checkpoints/"
inception_dir=scratch_dir+"rldraw/inception/"
gen_imgs_dir="./gen_imgs/"

import os
for d in [mnist_dir,checkpoint_dir,gen_imgs_dir,inception_dir]:
    if not os.path.exists(d):
        os.makedirs(d)