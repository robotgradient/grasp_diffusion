import torch

from se3dif.models.loader import load_model
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
import configargparse

def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=str, default='15')
    p.add_argument('--n_grasps', type=str, default='20')

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(obj_id=0):
    model_params = 'graspdif_model'
    batch = 15
    device = 'cpu'
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    context = torch.Tensor([[obj_id]]).to(device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    #generator = ApproximatedGrasp_AnnealedLD(model, batch=batch, T = 30, T_fit=50)
    generator = Grasp_AnnealedLD(model, batch=batch, T = 30, T_fit=20, k_steps=2, device=device)

    #generator.set_latent_code(obj_id)
    return generator, model


if __name__ == '__main__':

    args = parse_args()

    print('##########################################################')
    print(args.obj_id)
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    n_envs = 30

    ## Set Model and Sample Generator ##
    generator, model = get_approximated_grasp_diffusion_field(obj_id)

    H = generator.sample()
    H[..., :3, -1] *=1/8.

    ## Visualize results ##
    from se3dif.datasets import AcronymGraspsDirectory
    from se3dif.visualization import grasp_visualization
    from se3dif.utils import to_numpy
    acronym_grasps = AcronymGraspsDirectory()
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

    vis_H = H.squeeze()
    grasp_visualization.visualize_grasps(to_numpy(H), mesh=mesh)

