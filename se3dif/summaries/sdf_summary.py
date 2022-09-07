import numpy as np
import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt



def sdf_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    coords = model_input['x_sdf']
    pred_sdf = info['sdf'].reshape(coords.shape[0], coords.shape[1],1)
    gt_sdf = ground_truth['sdf'][0,:].squeeze().cpu().numpy()

    ## Occupancy Max-Min ##
    writer.add_scalar(prefix + "out_min", pred_sdf.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_sdf.max(), iter)
    writer.add_scalar(prefix + "target_out_min", gt_sdf.min(), iter)
    writer.add_scalar(prefix + "target_out_max", gt_sdf.max(), iter)

    ## Set colors based on good occupancy predictions ##
    input_coords = coords[:1].detach().cpu().numpy()
    pred_sdf = pred_sdf[:1,...].detach().cpu().numpy()

    all_colors = np.zeros_like(input_coords)

    pred_sdf = pred_sdf.squeeze()
    def set_color(all_colors, l_thrs=0., h_thrs=0.1, i=1, intensity=200):
        idxs = np.argwhere((pred_sdf<h_thrs) & (pred_sdf>l_thrs))
        color = np.zeros((1,3))
        color[0,i] = intensity

        all_colors[:, idxs,...] = color
        return all_colors
    all_colors = set_color(all_colors, h_thrs=0.05, i=0)
    all_colors = set_color(all_colors, l_thrs=0.05, h_thrs=0.1, i=1)
    all_colors = set_color(all_colors, l_thrs=0.1, h_thrs=0.3, i=1, intensity=100)
    all_colors = set_color(all_colors, l_thrs=0.3, h_thrs=0.5, i=2)


    point_cloud(writer, iter, prefix+'_colorized_sdf', input_coords, colors=all_colors)

    thrs = 0.07
    idxs = np.argwhere((pred_sdf < thrs))
    input_red = input_coords[:,idxs[:,0],...]
    point_cloud(writer, iter, prefix+'reconstr_sdf', input_red)

    idxs = np.argwhere((gt_sdf < thrs))
    input_red = input_coords[:,idxs[:,0],...]
    point_cloud(writer, iter, prefix+'reconstr_sdf_target', input_red)


def point_cloud(writer, iter, name, points_xyz, colors=None):
    point_size_config = {
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.05
        }
    }

    if colors is None:
       colors = np.zeros_like(points_xyz)

    writer.add_mesh(name, vertices=points_xyz, colors=colors,
                     config_dict={"material": point_size_config}, global_step=iter)

