import os
import numpy as np
import torch

base_dir = os.path.abspath(os.path.dirname(__file__))
pts_dir = os.path.join(base_dir)

def get_3d_pts(file=os.path.join(pts_dir,'UniformPts.npy') , scale = np.ones(3), loc = np.zeros(3), n_points=100):
    pts = np.load(file)
    pts = pts[:n_points,:]*scale + loc
    return torch.Tensor(pts)
