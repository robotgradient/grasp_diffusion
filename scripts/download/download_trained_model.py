import os
from se3dif.utils import makedirs
from huggingface_hub import hf_hub_url, hf_hub_download
import joblib

## Create model saving folder ##
base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
model_dir = os.path.join(base_dir, 'models','p_mugs')
makedirs(model_dir)

## Download models ##
repo_id = "camusean/grasp_diffusion"
model_filename = "model.pth"
params_filename = 'params.json'

hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=model_dir)
hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=model_dir)

print('model downloaded')