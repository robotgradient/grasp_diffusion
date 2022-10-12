import os
from moviepy.editor import VideoFileClip

image_folder = os.path.abspath(os.path.dirname(__file__))
video_name = os.path.join(image_folder, 'gifs/grasp_dif.mp4')


video = VideoFileClip(video_name)
video.write_gif("grasp_dif.gif", fps=video.fps/5)
