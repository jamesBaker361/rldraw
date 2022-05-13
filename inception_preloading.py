from pytorch_fid.inception import InceptionV3
import torch
import numpy as np
import argparse
from string_globals import *
from data_loading import *
from torch.nn.functional import adaptive_avg_pool2d

#inspired by https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

parser = argparse.ArgumentParser()

parser.add_argument("--dims", type=int, default=64,help="dims of features to extract, in 64, 192, 768, 2048")
parser.add_argument("--path",type=str, default=None, help="path to save output")
parser.add_argument("--char", type=int,default=0, help="what char 0-9 to calculate the statistics for")
parser.add_argument(
	"--batch_size", type=int, default=32, help="batch size for getting inception distance"
)

parser.add_argument(
	"--image_size", type=int, default=64, help="image l,w"
)

args = parser.parse_args()

if args.path is None:
    path=inception_dir+"size{}dims{}char{}".format(args.image_size,args.dims,args.char)
else:
    path=inception_dir+args.path
block_idx=InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]

dataloader,_=get_data_loaders_specific_char(mnist_dir,args.image_size,args.batch_size,args.char,3)
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
model = InceptionV3([block_idx]).to(device)

def get_activations(model,dataloader):
    pred_arr = np.empty((len(dataloader.dataset), args.dims))
    start_idx=0
    for batch in dataloader:
        batch=batch[0]
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_activation_statistics(pred_arr):
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


pred_arr=get_activations(model,dataloader)
mu,sigma=calculate_activation_statistics(pred_arr)
np.savez(path,mu=mu,sigma=sigma)
print("all done :), saved at {}".format(path))