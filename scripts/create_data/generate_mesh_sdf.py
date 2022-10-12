import os
import pickle
import h5py
import shutil
from mesh_to_sdf import sample_sdf_near_surface, get_surface_point_cloud

import trimesh
import logging
logging.getLogger("trimesh").setLevel(9000)
import numpy as np
from sklearn.neighbors import KDTree
import math
import pyrender


from se3dif.utils import makedirs

DATA_FOLDER = 'data'
OBJ_CLASSES = ['Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']

#OBJ_CLASSES = ['Bottle']
## Set data folder
base_folder = os.path.abspath(os.path.dirname(__file__)+'/..')
root_folder = os.path.abspath(os.path.join(base_folder, '..'))
data_folder = os.path.join(root_folder, DATA_FOLDER)
grasps_folder = os.path.join(data_folder, 'grasps')
meshes_folder = os.path.join(data_folder, 'meshes')
sdf_folder = os.path.join(data_folder, 'sdf')
makedirs(sdf_folder)


## Copied from mesh_to_sdf
def get_unit_spherize_scale(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    return np.max(distances)



def generate_mesh_sdf(mesh, absolute=True, normalize=False, n_points=200000):

    print (mesh)
    q_sdf, pcl = sample_sdf_near_surface(mesh, number_of_points=n_points, return_gradients=False)
    query_points, sdf = q_sdf[0], q_sdf[1]

    if absolute:
        neg_sdf_idxs = np.argwhere(sdf<0)[:,0]
        sdf[neg_sdf_idxs] = -sdf[neg_sdf_idxs]

    if normalize:
        sdf_max = sdf.max()
        sdf_min = sdf.min()
        sdf = (sdf - sdf_min) / (sdf_max - sdf_min)

    return query_points, sdf


if __name__ == '__main__':

    for obj_cls in OBJ_CLASSES:
        grasp_cls_folder = os.path.join(grasps_folder, obj_cls)
        count = 0
        for filename in os.listdir(grasp_cls_folder):
            try:
                count+=1
                print(count)
                ## Load Acronym file
                load_file = os.path.join(grasp_cls_folder, filename)
                print(filename)
                data = h5py.File(load_file, "r")
                ## Load mesh
                mesh_fname = data["object/file"][()].decode('utf-8')
                mesh_load_file = os.path.join(data_folder, mesh_fname)
                mesh = trimesh.load(mesh_load_file)
                scale = data["object/scale"][()]

                if type(mesh) == trimesh.scene.scene.Scene:
                    mesh = trimesh.util.concatenate(mesh.dump())

                scale = mesh.scale
                mesh.apply_scale(1/scale)
                H = np.eye(4)
                loc = mesh.centroid
                H[:-1, -1] = -loc
                mesh.apply_transform(H)


                print(mesh)


                mesh_name = mesh_fname.split('/')[-1]
                mesh_type = mesh_fname.split('/')[1]


                query_points, sdf = generate_mesh_sdf(mesh)

                ## save info
                save_sdf_folder = os.path.join(sdf_folder, mesh_type)
                makedirs(save_sdf_folder)


                sdf_mesh = mesh_name.split('.obj')[0] + '.json'
                save_file = os.path.join(save_sdf_folder, sdf_mesh)
                sdf_dict = {
                    'loc': loc,
                    'scale': scale,
                    'xyz': query_points,
                    'sdf': sdf,
                }

                with open(save_file, 'wb') as handle:
                    pickle.dump(sdf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


                ## VISUALIZE
                view_3d = False
                if view_3d:

                    colors = np.zeros(query_points.shape)
                    colors[:, 0] = sdf / 0.1 * (sdf < 0.1)
                    colors[:, 1] = (sdf - 0.1) / 0.6 * ((sdf > 0.1) & (sdf < 0.6))
                    colors[:, 2] = sdf * (sdf > 0.6)

                    idxs = np.argwhere(sdf < 10.01)[:, 0]
                    xyz = query_points[idxs, ...]
                    colors_xyz = colors[idxs, ...]

                    cloud = pyrender.Mesh.from_points(xyz, colors=colors_xyz)


                    scene = pyrender.Scene()

                    scene.add(cloud)
                    #scene.add(cloud_pcl)

                    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

            except:
                print('here')
