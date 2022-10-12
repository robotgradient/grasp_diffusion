import os
import glob
from PIL import Image
from natsort import natsorted # pip install natsort


image_folder = os.path.abspath(os.path.dirname(__file__))


# filepaths
fp_in = os.path.join(image_folder,'*.png')
fp_out = os.path.join(image_folder,'grasp_diffusion.gif')


imgs = (Image.open(f) for f in natsorted(glob.glob(fp_in)))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=100, loop=0)

print('gif made!')