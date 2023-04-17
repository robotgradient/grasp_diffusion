# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']

def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=0)
    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--obj_class', type=str, default='Laptop')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--eval_sim', type=bool, default=False)
    p.add_argument('--model', type=str, default='grasp_dif_multi')


    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, args, device='cpu'):
    model_params = args.model
    batch = 10
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    context = to_torch(p[None,...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

    return generator, model


def sample_pointcloud(obj_id=0, obj_class='Mug'):
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

    P = mesh.sample(1000)

    sampled_rot = scipy.spatial.transform.Rotation.random()
    rot = sampled_rot.as_matrix()
    rot_quat = sampled_rot.as_quat()

    P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)
    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)
    translational_shift = copy.deepcopy(H)



    return P, mesh, translational_shift, rot_quat


if __name__ == '__main__':
    import copy
    import configargparse
    args = parse_args()

    EVAL_SIMULATION = args.eval_sim
    # isaac gym has to be imported here as it is supposed to be imported before torch
    if (EVAL_SIMULATION):
        # Alternatively: Evaluate Grasps in Simulation:
        from isaac_evaluation.grasp_quality_evaluation import GraspSuccessEvaluator

    import scipy.spatial.transform
    import numpy as np
    from se3dif.datasets import AcronymGraspsDirectory
    from se3dif.models.loader import load_model
    from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
    from se3dif.utils import to_numpy, to_torch

    import torch



    print('##########################################################')
    print('Object Class: {}'.format(args.obj_class))
    print(args.obj_id)
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    n_envs = 30
    device = args.device

    ## Set Model and Sample Generator ##
    P, mesh, trans, rot_quad = sample_pointcloud(obj_id, obj_class)
    generator, model = get_approximated_grasp_diffusion_field(P, args, device)

    H = generator.sample()

    H_grasp = copy.deepcopy(H)
    # counteract the translational shift of the pointcloud (as the spawned model in simulation will still have it)
    H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()
    H[..., :3, -1] *=1/8.
    H_grasp[..., :3, -1] *=1/8.

    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H.squeeze()
    P *=1/8
    mesh = mesh.apply_scale(1/8)
    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P, mesh=mesh)

    if (EVAL_SIMULATION):
        ## Evaluate Grasps in Simulation##
        num_eval_envs = 10
        evaluator = GraspSuccessEvaluator(obj_class, n_envs=num_eval_envs, idxs=[args.obj_id] * num_eval_envs, viewer=True, device=device, \
                                          rotations=[rot_quad]*num_eval_envs, enable_rel_trafo=False)
        succes_rate = evaluator.eval_set_of_grasps(H_grasp)
        print('Success cases : {}'.format(succes_rate))
