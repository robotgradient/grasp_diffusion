import os, os.path as osp


## Set root directory
base_dir = os.path.abspath(os.path.dirname(__file__)+'../../../')
## Set root directory
root_directory = os.path.abspath(os.path.join(base_dir, '..'))
## Set data directory
data_directory = os.path.abspath(os.path.join(root_directory, 'data'))
# Set directory for meshes regarding the simulation environment:
mesh_dir = os.path.join(base_dir, 'isaac_evaluation', 'grasp_sim', 'meshes')

def get_pretrained_models_src():
    directory = osp.join(data_directory,'models')
    makedirs(directory)
    return directory


def get_data_src():
    makedirs(data_directory)
    return data_directory


def get_grasps_src():
    directory = osp.join(get_data_src(), 'grasps')
    return directory


def get_root_src():
    return root_directory


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_mesh_src():
    return mesh_dir

