import glob
import copy
import time

import numpy as np
import trimesh
import open3d as o3d

from scipy.stats import special_ortho_group
import functools

import os
import torch

from torch.utils.data import DataLoader, Dataset
import json
import pickle
import h5py
from se3dif.utils import get_data_src

from se3dif.utils import to_numpy, to_torch, get_grasps_src
from mesh_to_sdf.surface_point_cloud import get_scan_view, get_hq_scan_view
from mesh_to_sdf.scan import ScanPointcloud
from pathlib import Path
from mesh_to_sdf import sample_sdf_near_surface
from se3dif.utils import directory_utils

import os, sys

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class AcronymGrasps:
    def __init__(self, path):
        base_dir: str = Path(path).parents[3]  # data/

        with h5py.File(path, 'r') as store:
            self.pose = store['grasp'][...][:, :16].reshape(-1, 4, 4)
            self.center = store.attrs['center']

            # required for se3 diffusion paper
            self.mesh_scale = store.attrs['scale']
            self.mesh_id = store.attrs['fname']
            self.mesh_type = str(path).split("/")[-3]
            self.mesh_fname = f"objs/{self.mesh_type}/train/{self.mesh_id}_visual.obj"

            self.grasps, self.success = self.load_grasps()
            good_idxs = np.argwhere(self.success == 1)[:, 0]
            bad_idxs = np.argwhere(self.success == 0)[:, 0]
            self.good_grasps = self.grasps[good_idxs, ...]
            self.bad_grasps = self.grasps[bad_idxs, ...]

    # def get_signed_distance(self, mesh: trimesh.Trimesh, samples: np.ndarray):
    def get_signed_distance(self, samples: np.ndarray):
        mesh = self.load_mesh()

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
        dist = scene.compute_signed_distance(o3d.core.Tensor.from_numpy(samples.astype(np.float32))).numpy()
        return dist

    def load_grasps(self):
        # T, success
        # pose(grasp) is all success
        return self.pose, np.ones(len(self.pose))

    def get_mesh(self, *args, **kwargs) -> trimesh.Trimesh:
        return self.load_mesh(*args, **kwargs)

    def load_mesh(self) -> trimesh.Trimesh:
        mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)

        mesh = trimesh.load(mesh_path_file,  file_type='obj', force='mesh')
        mesh.apply_translation(-self.center)
        mesh.apply_scale(self.mesh_scale)

        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())

        return mesh

class AcronymGraspsDirectory():
    def __init__(self, filename=get_grasps_src(), data_type='Mug'):
        grasps_files = sorted(glob.glob(
            os.path.join(filename, data_type, 'train/*.h5')
        ))

        self.avail_obj = []
        for grasp_file in grasps_files:
            self.avail_obj.append(AcronymGrasps(grasp_file))


class AcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF Auto-Decoder model'
    def __init__(self, class_type='Mug', se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1500,
                 augmented_rotation=True, visualize=False, split = True):

        raise NotImplementedError("AcronymAndSDFDataset Not Implemented")
        self.class_type = class_type
        self.data_dir = get_data_src()
        self.acronym_data_dir = self.data_dir

        self.grasps_dir = os.path.join(self.acronym_data_dir, 'grasps')
        self.sdf_dir = os.path.join(self.acronym_data_dir, 'sdf')

        self.generated_points_dir = os.path.join(self.acronym_data_dir, 'train_data')

        grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type+'/*.h5'))

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
    def __init__(self, class_type=['mug'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000,
                 augmented_rotation=True, visualize=False, split = True):

        #class_type = ['Mug']
        self.class_type = class_type
        self.data_dir = get_data_src()

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')
        self.meshes_dir = os.path.join(self.data_dir, 'objs')

        self.grasp_files = []
        for class_type_i in class_type:
            ## Get Grasp File
            cls_grasps_files = sorted(glob.glob(
                os.path.join(self.grasps_dir, class_type_i, 'train/*.h5')
            ))

            for grasp_file in cls_grasps_files:
                g_obj = AcronymGrasps(grasp_file)

                ## Grasp File ##
                if g_obj.good_grasps.shape[0] > 0:
                    self.grasp_files.append(grasp_file)

        ## Split Train/Validation
        n = len(self.grasp_files)
        train_size = int(n*0.9)
        test_size  =  n - train_size

        self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(self.grasp_files, [train_size, test_size])

        self.type = 'train'
        self.len = len(self.train_grasp_files)

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

    def set_test_data(self):
        self.len = len(self.test_grasp_files)
        self.type = 'test'

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
        except:
            print('lets see')
        H_grasps = grasp_obj.good_grasps[rix, ...]
        return H_grasps

    def _get_sdf(self, grasp_obj, grasp_file):

        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_type = mesh_fname.split('/')[1]
        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(
            directory_utils.get_sdf_src(),
            mesh_type,
            filename+'.json',
        )

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale

        # (loc, scale) because applied when sdf is generated
        # grasps obj AcrGrasp
        # scale: obj file scale
        # mesh_scale: grasps obj .h5 file
        return xyz, sdf


    def _get_sample_sdf_points(self, grasp_obj):
        case = 0

        if case == 0:
            mesh = grasp_obj.load_mesh()

            # NOTE: original code set it as 200000
            q_sdf, pcl = sample_sdf_near_surface(mesh, number_of_points=1000)
            query_points = q_sdf[0]
        else:
            query_points = np.random.rand(1000, 3)

        return query_points

    def _get_sdf_deprecated(self, grasp_obj, grasp_file):
        query_points = self._get_sample_sdf_points(grasp_obj)

        xyz = query_points
        sdf = grasp_obj.get_signed_distance(query_points)

        mesh_scale = grasp_obj.mesh_scale

        xyz = xyz * mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf[rix[:self.n_occ]] * mesh_scale

        return xyz, sdf

    def _get_mesh_pcl(self, grasp_obj):
        mesh = grasp_obj.load_mesh()
        return mesh.sample(self.n_pointcloud)

    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = AcronymGrasps(self.train_grasp_files[index])
        else:
            grasps_obj = AcronymGrasps(self.test_grasp_files[index])

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)

        ## rescale, rotate and translate ##
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
        #######################

        # Visualize
        if self.visualize:

            ## 3D matplotlib ##
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            ## sdf visualization ##
            n = 100
            x = xyz[:n,:]

            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show()
            #plt.show(block=True)

        res = {'visual_context': torch.from_numpy(pcl).float(),
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}

        return res, {'sdf': torch.from_numpy(sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


class PartialPointcloudAcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000,
                 augmented_rotation=True, visualize=False, split = True, test_files=None):
        raise NotImplementedError("PPCL Not Implemented")
        self.class_type = class_type
        self.data_dir = get_data_src()

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')

        self.grasp_files = []
        for class_type_i in class_type:
            cls_grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type_i+'/*.h5'))

            for grasp_file in cls_grasps_files:
                g_obj = AcronymGrasps(grasp_file)

                ## Grasp File ##
                if g_obj.good_grasps.shape[0] > 0:
                    self.grasp_files.append(grasp_file)

        ## Split Train/Validation
        n = len(self.grasp_files)
        train_size = int(n*0.9)
        test_size  =  n - train_size

        self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(self.grasp_files, [train_size, test_size])
        self.type = 'train'
        self.len = len(self.train_grasp_files)

        if test_files is not None:
            self.test_grasp_files = test_files
            self.set_test_data()

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

        ## Sampler
        self.scan_pointcloud = ScanPointcloud()

    def __len__(self):
        return self.len

    def set_test_data(self):
        self.len = len(self.test_grasp_files)
        self.type = 'test'

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
        except:
            print('lets see')
        H_grasps = grasp_obj.good_grasps[rix, ...]
        return H_grasps

    def _get_sdf(self, grasp_obj, grasp_file):

        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_type = mesh_fname.split('/')[1]
        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename+'.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale
        return xyz, sdf

    def _get_mesh_pcl(self, grasp_obj):
        mesh = grasp_obj.load_mesh()
        ## 1. Mesh Centroid ##
        centroid = mesh.centroid
        H = np.eye(4)
        H[:3, -1] = -centroid
        mesh.apply_transform(H)
        ######################
        #time0 = time.time()
        P = self.scan_pointcloud.get_hq_scan_view(mesh)
        #print('Sample takes {} s'.format(time.time() - time0))
        P +=centroid
        try:
            rix = np.random.randint(low=0, high=P.shape[0], size=self.n_pointcloud)
        except:
            print('here')
        return P[rix, :]

    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = AcronymGrasps(self.train_grasp_files[index])
        else:
            grasps_obj = AcronymGrasps(self.test_grasp_files[index])

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)

        ## rescale, rotate and translate ##
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
        #######################

        # Visualize
        if self.visualize:

            ## 3D matplotlib ##
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            ## sdf visualization ##
            n = 100
            x = xyz[:n,:]

            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show()
            #plt.show(block=True)

        res = {'visual_context': torch.from_numpy(pcl).float(),
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}

        return res, {'sdf': torch.from_numpy(sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


if __name__ == '__main__':

    ## Index conditioned dataset
    dataset = AcronymAndSDFDataset(visualize=True, augmented_rotation=True, one_object=False)

    ## Pointcloud conditioned dataset
    dataset = PointcloudAcronymAndSDFDataset(visualize=True, augmented_rotation=True, one_object=False)

    ## Pointcloud conditioned dataset
    dataset = PartialPointcloudAcronymAndSDFDataset(visualize=False, augmented_rotation=True, one_object=False)

    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for x,y in train_dataloader:
        print(x)
