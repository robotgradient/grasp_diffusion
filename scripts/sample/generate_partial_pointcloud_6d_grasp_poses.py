import configargparse
import torch

import numpy as np
import scipy.spatial.transform as T

from mesh_to_sdf.scan import ScanPointcloud
from se3dif.datasets import AcronymGraspsDirectory
from se3dif.models.loader import load_model
from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch


def parse_args():
    p = configargparse.ArgumentParser()
    p.add(
        "-c",
        "--config_filepath",
        required=False,
        is_config_file=True,
        help="Path to config file.",
    )

    p.add_argument("--obj_id", type=str, default="12")
    p.add_argument("--n_grasps", type=str, default="20")
    p.add_argument("--obj_class", type=str, default="Mug")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=20)

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, batch_size=10, device="cpu"):
    model_params = "partial_grasp_dif"

    ## Load model
    model_args = {"device": device, "pretrained_model": model_params}
    model = load_model(model_args)

    context = to_torch(p[None, ...], device)
    model.set_latent(context, batch_size)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(
        model, device=device, batch=batch_size, T=70, T_fit=50, k_steps=1
    )

    # generator.set_latent_code(obj_id)
    return generator, model


def sample_pointcloud(obj_id=0, obj_class="Mug"):
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

    centroid = mesh.centroid
    H = np.eye(4)
    H[:3, -1] = -centroid
    mesh.apply_transform(H)

    scan_pointcloud = ScanPointcloud()
    P = scan_pointcloud.get_hq_scan_view(mesh)

    P *= 8.0
    P_mean = np.mean(P, 0)
    P += -P_mean

    rot = T.Rotation.random().as_matrix()
    P = np.einsum("mn,bn->bm", rot, P)

    mesh.apply_scale(8.0)
    H = np.eye(4)
    H[:3, -1] = -P_mean
    mesh.apply_transform(H)
    H = np.eye(4)
    H[:3, :3] = rot
    mesh.apply_transform(H)

    return P, mesh


if __name__ == "__main__":

    args = parse_args()

    print("##########################################################")
    print("Object Class: {}".format(args.obj_class))
    print(args.obj_id)
    print("##########################################################")

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    device = args.device
    batch_size = args.batch_size

    ## Set Model and Sample Generator ##
    P, mesh = sample_pointcloud(obj_id, obj_class)
    generator, model = get_approximated_grasp_diffusion_field(
        P, batch_size=batch_size, device=device
    )

    H_batches = []
    batches = int(np.ceil((n_grasps / batch_size)))
    for i in range(0, batches):
        H_batches.append(generator.sample())

    H = torch.concatenate(H_batches, 0)
    H[..., :3, -1] *= 1 / 8.0

    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H.squeeze()
    P *= 1 / 8
    mesh = mesh.apply_scale(1 / 8)
    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P)  # , mesh=mesh)
