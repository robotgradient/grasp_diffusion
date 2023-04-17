import glob
import copy
import time

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
from mesh_to_sdf.surface_point_cloud import get_scan_view, get_hq_scan_view
from mesh_to_sdf.scan import ScanPointcloud



import os, sys

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


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
        mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)

        mesh = trimesh.load(mesh_path_file,  file_type='obj', force='mesh')

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh


class AcronymGraspsDirectory():
    def __init__(self, filename=get_grasps_src(), data_type='Mug'):

        grasps_files = sorted(glob.glob(filename + '/' + data_type + '/*.h5'))

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
    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
                                   'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
                                   'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
                                   'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
                                   'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
                                   'ToyFigure', 'Wallet','WineGlass',
                                   'Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000,
                 augmented_rotation=True, visualize=False, split = True):

        #class_type = ['Mug']
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
