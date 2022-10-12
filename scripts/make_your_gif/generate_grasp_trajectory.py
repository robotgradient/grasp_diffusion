import os
import scipy.spatial.transform
import torch
import numpy as np
from se3dif.datasets import AcronymGraspsDirectory
from se3dif.models.loader import load_model
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch
import configargparse

image_save_folder = os.path.abspath(os.path.dirname(__file__))
device = 'cpu'

N_GRASPS = 8
OBJ_CLASS = 'Bowl'
OBJ_ID = '4'

def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=str, default=OBJ_ID)
    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--obj_class', type=str, default=OBJ_CLASS)

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, device='cpu'):
    model_params = 'multiobject_p_graspdif2'
    batch = N_GRASPS
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    context = to_torch(p[None,...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=100, k_steps=1)

    #generator.set_latent_code(obj_id)
    return generator, model


def sample_pointcloud(obj_id=0, obj_class='Mug'):
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

    P = mesh.sample(1000)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    rot = scipy.spatial.transform.Rotation.random().as_matrix()
    rot = np.eye(3)
    P = np.einsum('mn,bn->bm', rot, P)

    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)
    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)

    return P, mesh


if __name__ == '__main__':

    args = parse_args()

    print('##########################################################')
    print('Object Class: {}'.format(args.obj_class))
    print(args.obj_id)
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    n_envs = 30

    ## Set Model and Sample Generator ##
    P, mesh = sample_pointcloud(obj_id, obj_class)
    generator, model = get_approximated_grasp_diffusion_field(P, device)

    H, trj_H = generator.sample(save_path=True)
    H[..., :3, -1] *=1/8.
    trj_H[..., :3, -1] *=1/8.


    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H.squeeze()
    P *=1/8
    mesh = mesh.apply_scale(1/8)

    T = torch.linspace(0, trj_H.shape[0]-1, 80).int()
    colors = np.random.rand(trj_H.shape[1], 3)*0.5 + 0.5

    for t in range(T.shape[0]):

        vis_H = trj_H[T[t],...]
        scene = grasp_visualization.visualize_grasps(to_numpy(vis_H), colors=colors, mesh=mesh, show=False)
        scene.set_camera(distance=.6, center=[0., 0., 0.], angles=[0.8, 0.0, 0.8])

        ## Save as .png
        filename = os.path.join(image_save_folder, 'fig_{}.png'.format(t))
        png = scene.save_image(resolution=[1028, 1028], visible=True)
        with open(filename, 'wb') as f:
            f.write(png)
            f.close()

    ## Generate GIF from images ##
    import os
    import glob
    from PIL import Image
    from natsort import natsorted # pip install natsort


    image_folder = os.path.abspath(os.path.dirname(__file__))


    # filepaths
    fp_in = os.path.join(image_folder,'*.png')
    fp_out = os.path.join(image_folder,'{}_{}.gif'.format(OBJ_CLASS,OBJ_ID))


    imgs = (Image.open(f) for f in natsorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)

    print('gif made!')


