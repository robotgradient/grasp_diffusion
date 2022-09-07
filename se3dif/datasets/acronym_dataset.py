import glob
import copy
import numpy as np
import trimesh

from scipy.stats import special_ortho_group

import os
import torch

from torch.utils.data import DataLoader, Dataset
import json
import pickle
import h5py
from se3dif.utils import get_data_src

from se3dif.utils import to_numpy, to_torch, get_grasps_src


class AcronymGrasps():
    def __init__(self, filename):

        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = data["object/file"][()].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.grasps, self.success = self.load_grasps(filename)
        good_idxs = np.argwhere(self.success==1)[:,0]
        bad_idxs  = np.argwhere(self.success==0)[:,0]
        self.good_grasps = self.grasps[good_idxs,...]
        self.bad_grasps  = self.grasps[bad_idxs,...]

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def load_mesh(self):
        mesh_dir = os.path.join(get_data_src(), 'acronym', 'meshes')
        try:
            mesh_path_dir = os.path.join(mesh_dir, self.mesh_type, self.mesh_id)
            mesh_path_file = os.path.join(mesh_path_dir, 'mesh.obj')
            mesh = trimesh.load_mesh(mesh_path_file)
        except:
            mesh_path_dir = os.path.join(mesh_dir, self.mesh_type, self.mesh_id+'.obj')
            mesh_path_file = os.path.join(mesh_path_dir, self.mesh_id+'.obj')
            mesh = trimesh.load_mesh(mesh_path_file)
        mesh.apply_scale(self.mesh_scale)
        return mesh


class AcronymGraspsDirectory():
    def __init__(self, filename=get_grasps_src(), data_type='Mug'):

        grasps_files = sorted(glob.glob(filename + '/' + data_type + '*.h5'))
        self.avail_obj = []
        for grasp_file in grasps_files:
            self.avail_obj.append(AcronymGrasps(grasp_file))


class AcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF Auto-Decoder model'
    def __init__(self, class_type='Mug', se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1500,
                 augmented_rotation=True, visualize=False, split = True):

        self.class_type = class_type
        self.data_dir = get_data_src()
        self.acronym_data_dir = os.path.join(self.data_dir, 'acronym')

        self.grasps_dir = os.path.join(self.acronym_data_dir, 'grasps')
        self.sdf_dir = os.path.join(self.acronym_data_dir, 'sdf')

        self.generated_points_dir = os.path.join(self.acronym_data_dir, 'train_data')

        grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type+'*.h5'))

        points_files = []
        sdf_files = []
        for grasp_file in grasps_files:
            g_obj = AcronymGrasps(grasp_file)
            mesh_file = g_obj.mesh_fname
            txt_split = mesh_file.split('/')

            sdf_file = os.path.join(self.sdf_dir, class_type, txt_split[-1].split('.')[0]+'.json')
            point_file = os.path.join(self.generated_points_dir, class_type, '4_points', txt_split[-1]+'.npz')

            sdf_files.append(sdf_file)
            points_files.append(point_file)

        ## Split Train/Validation
        n = len(grasps_files)
        indexes = np.arange(0, n)
        self.total_len = n
        if split:
            idx = int(0.9 * n)
        else:
            idx = int(n)

        if phase == 'train':
            self.grasp_files = grasps_files[:idx]
            self.points_files = points_files[:idx]
            self.sdf_files = sdf_files[:idx]
            self.indexes = indexes[:idx]
        else:
            self.grasp_files = grasps_files[idx:]
            self.points_files = points_files[idx:]
            self.sdf_files = sdf_files[idx:]
            self.indexes = indexes[idx:]


        self.len = len(self.points_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

    def __len__(self):
        return self.len

    def _get_item(self, index):
        if self.one_object:
            index = 0

        index_obj = self.indexes[index]
        ## Load Files ##
        grasps_obj = AcronymGrasps(self.grasp_files[index])
        sdf_file = self.sdf_files[index]
        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        ## PointCloud
        p_clouds = sdf_dict['pcl']
        rix = np.random.permutation(p_clouds.shape[0])
        p_clouds = p_clouds[rix[:self.n_pointcloud],:]

        ## Coordinates XYZ
        coords  = sdf_dict['xyz']
        rix = np.random.permutation(coords.shape[0])
        coords = coords[rix[:self.n_occ],:]

        ### SDF value
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]
        grad_sdf = sdf_dict['grad_sdf'][rix[:self.n_occ], ...]

        ### Scale and Loc
        scale = sdf_dict['scale']
        loc = sdf_dict['loc']

        ## Grasps good/bad
        rix = np.random.randint(low=0, high=grasps_obj.good_grasps.shape[0], size=self.n_density)
        H_grasps = grasps_obj.good_grasps[rix, ...]
        rix = np.random.randint(low=0, high=grasps_obj.bad_grasps.shape[0], size=self.n_density)
        H_bad_grasps = grasps_obj.bad_grasps[rix, ...]

        ## Rescale Pointcloud and Occupancy Points ##
        coords = (coords + loc)*scale*grasps_obj.mesh_scale * self.scale
        p_clouds = (p_clouds + loc)*scale*grasps_obj.mesh_scale * self.scale

        sdf = sdf*scale*grasps_obj.mesh_scale * self.scale
        grad_sdf = -grad_sdf*scale*grasps_obj.mesh_scale * self.scale

        H_grasps[:,:-1,-1] = H_grasps[:,:-1,-1] * self.scale
        H_bad_grasps[:,:-1,-1] = H_bad_grasps[:,:-1,-1]*self.scale

        ## Random rotation ##
        if self.augmented_rotation:
            R = special_ortho_group.rvs(3)
            H = np.eye(4)
            H[:3,:3] = R

            coords = np.einsum('mn,bn->bm',R, coords)
            p_clouds = np.einsum('mn,bn->bm',R, p_clouds)

            H_grasps = np.einsum('mn,bnd->bmd', H, H_grasps)
            H_bad_grasps = np.einsum('mn,bnd->bmd', H, H_bad_grasps)

            grad_sdf = np.einsum('mn,bn->bm', R, grad_sdf)


        # Visualize
        if self.visualize:
            ## 3D matplotlib ##
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(p_clouds[:,0], p_clouds[:,1], p_clouds[:,2], c='r')

            n = 10
            x = coords[:n,:]
            ## grad sdf ##
            x_grad = grad_sdf[:n, :]
            ax.quiver(x[:,0], x[:,1], x[:,2], x_grad[:,0], x_grad[:,1], x_grad[:,2], length=0.3)

            ## sdf visualization ##
            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show(block=True)

        del sdf_dict

        res = {'point_cloud': torch.from_numpy(p_clouds).float(),
               'x_sdf': torch.from_numpy(coords).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'x_neg_ene': torch.from_numpy(H_bad_grasps).float(),
               'scale': torch.Tensor([self.scale]).float(),
               'visual_context':  torch.Tensor([index_obj])}

        return res, {'sdf': torch.from_numpy(sdf).float(), 'grad_sdf': torch.from_numpy(grad_sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


class PointcloudAcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, class_type='Mug', se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000,
                 augmented_rotation=True, visualize=False, split = True):

        self.class_type = class_type
        self.data_dir = get_data_src()
        self.acronym_data_dir = os.path.join(self.data_dir, 'acronym')

        self.grasps_dir = os.path.join(self.acronym_data_dir, 'grasps')
        self.sdf_dir = os.path.join(self.acronym_data_dir, 'sdf')

        self.generated_points_dir = os.path.join(self.acronym_data_dir, 'train_data')

        grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type+'*.h5'))

        points_files = []
        sdf_files = []
        for grasp_file in grasps_files:
            g_obj = AcronymGrasps(grasp_file)
            mesh_file = g_obj.mesh_fname
            txt_split = mesh_file.split('/')

            sdf_file = os.path.join(self.sdf_dir, class_type, txt_split[-1].split('.')[0]+'.json')
            point_file = os.path.join(self.generated_points_dir, class_type, '4_points', txt_split[-1]+'.npz')

            sdf_files.append(sdf_file)
            points_files.append(point_file)

        ## Split Train/Validation
        n = len(grasps_files)
        indexes = np.arange(0, n)
        self.total_len = n
        if split:
            idx = int(0.99 * n)
        else:
            idx = int(n)

        if phase == 'train':
            self.grasp_files = grasps_files[:idx]
            self.points_files = points_files[:idx]
            self.sdf_files = sdf_files[:idx]
            self.indexes = indexes[:idx]
        else:
            self.grasp_files = grasps_files[idx:]
            self.points_files = points_files[idx:]
            self.sdf_files = sdf_files[idx:]
            self.indexes = indexes[idx:]


        self.len = len(self.points_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

    def __len__(self):
        return self.len

    def _get_item(self, index):
        if self.one_object:
            index = 0

        index_obj = self.indexes[index]
        ## Load Files ##
        grasps_obj = AcronymGrasps(self.grasp_files[index])
        sdf_file = self.sdf_files[index]
        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        ## PointCloud
        p_clouds_all = sdf_dict['pcl']
        rix = np.random.permutation(p_clouds_all.shape[0])
        self.n_pointcloud = 1000
        p_clouds = p_clouds_all[rix[:self.n_pointcloud],:]

        ## Test ##
        # mesh = grasps_obj.load_mesh()
        #
        # p_test = mesh.sample(1000)
        #
        # p_test = p_test*self.scale
        # p_test_mean = np.mean(p_test, 0)
        # p_test = p_test - p_test_mean
        # print('p_test_mean: {}'.format(p_test_mean))


        ## Coordinates XYZ
        coords  = sdf_dict['xyz']
        rix = np.random.permutation(coords.shape[0])
        coords = coords[rix[:self.n_occ],:]

        ### SDF value
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]
        grad_sdf = sdf_dict['grad_sdf'][rix[:self.n_occ], ...]

        ### Scale and Loc
        scale = sdf_dict['scale']
        loc = sdf_dict['loc']

        ## Grasps good/bad
        rix = np.random.randint(low=0, high=grasps_obj.good_grasps.shape[0], size=self.n_density)
        H_grasps = grasps_obj.good_grasps[rix, ...]
        rix = np.random.randint(low=0, high=grasps_obj.bad_grasps.shape[0], size=self.n_density)
        H_bad_grasps = grasps_obj.bad_grasps[rix, ...]

        ## Rescale Pointcloud and Occupancy Points ##
        coords = (coords + loc)*scale*grasps_obj.mesh_scale * self.scale
        p_clouds = (p_clouds + loc)*scale*grasps_obj.mesh_scale * self.scale

        sdf = sdf*scale*grasps_obj.mesh_scale * self.scale
        grad_sdf = -grad_sdf*scale*grasps_obj.mesh_scale * self.scale

        H_grasps[:,:-1,-1] = H_grasps[:,:-1,-1] * self.scale
        H_bad_grasps[:,:-1,-1] = H_bad_grasps[:,:-1,-1]*self.scale

        ## Random rotation ##

        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3,:3] = R

        ## Center and Rotate ##
        p_mean = np.mean(p_clouds, 0)
        if self.augmented_rotation:
            p_clouds = p_clouds - p_mean
            p_clouds = np.einsum('mn,bn->bm',R, p_clouds)

            coords = coords - p_mean
            coords = np.einsum('mn,bn->bm',R, coords)

            H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - p_mean
            H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)

            H_bad_grasps[..., :3, -1] = H_bad_grasps[..., :3, -1] - p_mean
            H_bad_grasps = np.einsum('mn,bnk->bmk', H, H_bad_grasps)

        #######################

        # Visualize
        if self.visualize:

            ## 3D matplotlib ##
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(p_clouds[:,0], p_clouds[:,1], p_clouds[:,2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            n = 10
            x = coords[:n,:]
            ## grad sdf ##
            x_grad = grad_sdf[:n, :]
            ax.quiver(x[:,0], x[:,1], x[:,2], x_grad[:,0], x_grad[:,1], x_grad[:,2], length=0.3)

            ## sdf visualization ##
            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show(block=True)

        del sdf_dict

        res = {'visual_context': torch.from_numpy(p_clouds).float(),
               'x_sdf': torch.from_numpy(coords).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'x_neg_ene': torch.from_numpy(H_bad_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}

        return res, {'sdf': torch.from_numpy(sdf).float(), 'grad_sdf': torch.from_numpy(grad_sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)



if __name__ == '__main__':

    ## Index conditioned dataset
    dataset = AcronymAndSDFDataset(visualize=True, augmented_rotation=True, one_object=False)

    ## Pointcloud conditioned dataset
    dataset = PointcloudAcronymAndSDFDataset(visualize=True, augmented_rotation=True, one_object=True)

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for x,y in train_dataloader:
        print(x)
