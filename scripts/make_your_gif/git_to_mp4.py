import os
import glob
from PIL import Image
from natsort import natsorted  # pip install natsort
import moviepy.editor as mp

image_folder = os.path.abspath(os.path.dirname(__file__))

# filepaths
fp_in = os.path.join(image_folder, '*.gif')
filenames = natsorted(glob.glob(fp_in))

for file in filenames:
    gif_name = file
    clip = mp.VideoFileClip(file)

    video_name = gif_name.split('.gif')[0] + '.mp4'
    clip.write_videofile(video_name)