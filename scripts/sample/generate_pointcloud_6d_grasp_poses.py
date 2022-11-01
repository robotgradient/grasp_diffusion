import configargparse
import torch

import numpy as np
import scipy.spatial.transform as T

from se3dif.datasets import AcronymGraspsDirectory
from se3dif.models.loader import load_model
from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch

# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']


def parse_args():
    p = configargparse.ArgumentParser()
    p.add(
        "-c",
        "--config_filepath",
        required=False,
        is_config_file=True,
        help="Path to config file.",
    )

    p.add_argument("--obj_id", type=str, default="0")
    p.add_argument("--n_grasps", type=str, default="200")
    p.add_argument("--obj_class", type=str, default="Laptop")
    p.add_argument("--model", type=str, default="point_graspdif")
    p.add_argument("--device", type=str, default="cpu")

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, args, batch_size=10, device="cpu"):
    model_params = args.model

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

    P = mesh.sample(1000)
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

    batch_size = 10

    ## Set Model and Sample Generator ##
    P, mesh = sample_pointcloud(obj_id, obj_class)
    generator, model = get_approximated_grasp_diffusion_field(
        P, args, batch_size=batch_size, device=device
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
    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P, mesh=mesh)
