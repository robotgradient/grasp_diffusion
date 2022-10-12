import os
import pickle
import h5py
import shutil

from se3dif.utils import makedirs

SAVE_FOLDER = 'data'

base_folder = os.path.abspath(os.path.dirname(__file__)+'/..')
root_folder = os.path.abspath(os.path.join(base_folder, '..'))

acronym_path = os.path.join(root_folder, 'source_data', 'grasps')
shapenet_path = os.path.join(root_folder, 'source_data', 'models-OBJ', 'models')

data_folder = os.path.join(root_folder, SAVE_FOLDER)
makedirs(data_folder)


for filename in os.listdir(acronym_path):
    print(filename)
    ## Load Acronym file
    load_file = os.path.join(acronym_path, filename)
    data = h5py.File(load_file, "r")

    ## check mesh
    mesh_fname = data["object/file"][()].decode('utf-8')
    mesh_name = mesh_fname.split('/')[-1]
    mesh_type = mesh_fname.split('/')[1]
    print(mesh_fname)

    ## Generate Mesh folder and copy file
    mesh_folder = os.path.dirname(mesh_fname)
    save_mesh_folder = os.path.join(data_folder, mesh_folder)
    makedirs(save_mesh_folder)

    src_mesh = os.path.join(shapenet_path, mesh_name)
    shutil.copy(src_mesh, save_mesh_folder)
    # mtl_mesh = src_mesh.split('.obj')[0]+'.mtl'
    # shutil.copy(mtl_mesh, save_mesh_folder)
    # jpg_mesh = src_mesh.split('.obj')[0]+'.jpg'
    # try:
    #     shutil.copy(jpg_mesh, save_mesh_folder)
    # except:
    #     pass


    ## Generate Grasp folder and copy file
    save_grasp_folder = os.path.join(data_folder, 'grasps', mesh_type)
    makedirs(save_grasp_folder)
    shutil.copy(load_file, save_grasp_folder)











