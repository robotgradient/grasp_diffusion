import numpy as np
import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


from se3dif.samplers import Grasp_AnnealedLD

from se3dif.utils import to_numpy
from se3dif.visualization import grasp_visualization


def denoising_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    observation = model_input['visual_context']
    batch = 4

    ## 1. visualize generated grasps ##
    model.eval()
    model.set_latent(observation[:1,...], batch=batch)
    generator = Grasp_AnnealedLD(model, batch=batch, T=30, T_fit=50, device=observation.device)
    H = generator.sample()

    H = to_numpy(H)
    H[:, :3, -1]*=1/8
    if observation.dim()==3:
        point_cloud = to_numpy(model_input['visual_context'])[0,...]/8.
    else:
        point_cloud = to_numpy(model_input['point_cloud'])[0,...]/8.

    image = grasp_visualization.get_scene_grasps_image(H, p_cloud=point_cloud)
    figure = plt.figure()
    plt.imshow(image)
    writer.add_figure("diffusion/generated_grasps", figure, global_step=iter)






