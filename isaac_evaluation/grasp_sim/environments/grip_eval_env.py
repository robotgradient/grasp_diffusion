from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
import math
import time
import os, os.path as osp
import copy
from se3dif.utils import directory_utils
from isaac_evaluation.utils.geometry_utils import Transform_2_H
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

from isaac_evaluation.grasp_sim.objects.object import SimpleObject
from isaac_evaluation.grasp_sim.robots.gripper_only import GripperOnly


class GraspingGymEnv():
    '''
    Environment to evaluate the Grasping of a certain object
    '''
    def __init__(self, gym, sim, env, isaac_base, curr_env_number, args=None):

        ## Set Args
        self.args = self._set_args(args)
        self.n_dofs = 16

        ## Set Hyperparams
        self.gym = gym
        self.sim = sim
        self.env = env
        self.isaac_base = isaac_base
        self.curr_env_number = curr_env_number

        ## Build Environment
        self._create_env()

    def _set_args(self, args):
        if args is None:
            obj_args = {
                'object_type':'rectangle',
                'object_id':'rectangle',
                'object_name':'rectangle',
                'scale': 1.,
            }
            args = {'obj_args':obj_args}
        else:
            args = args

        if 'obj_or' not in args['obj_args']:
            args['obj_args']['obj_or'] = np.array([0., 0., 0., 1.])

        return args

    def _create_env(self):
        self.table = self._load_table()
        self.obj, self.initial_obj_pose = self._load_obj(self.args['obj_args'])
        self.gripper = self._load_gripper(self.initial_obj_pose)

    def _load_table(self):
        # create table
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.use_mesh_materials = True
        table_path = osp.join(directory_utils.get_mesh_src(), 'table/table.urdf')
        table_asset = self.gym.load_asset(
            self.sim, '', table_path, asset_options)

        # table pose:
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, -0.02)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_handle = self.gym.create_actor(self.env, table_asset, table_pose, "table", self.curr_env_number)
        return table_handle

    def _load_obj(self, args):

        obj_type = args['object_type']
        obj_id   = args['object_id']
        obj_name = args['object_name']
        scale = args['scale']

        #obj_ori = args['obj_ori']
        quat = args['obj_or']

        # create new shape object:
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0.0, 0.0, 0.9)

        obj_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

        self.shape_obj = SimpleObject(self.gym, self.sim, self.env, self.isaac_base, self.curr_env_number, obj_pose, obj_type=obj_type,
                                     obj_id = obj_id, obj_name = obj_name, scale=scale)
        return self.shape_obj, obj_pose

    def _load_gripper(self, obj_pose):

        pose = gymapi.Transform()
        ## Compute initial rotation
        T_grasp = np.eye(4)
        Rot = R.from_euler('x', 180, degrees=True).as_matrix()
        T_grasp[:3, :3] = Rot
        grasp_trans = T_grasp[:3,-1]
        grasp_quat = R.from_matrix(T_grasp[:3,:3]).as_quat()
        ## Compute initial position
        pose.p = obj_pose.p
        pose.p.z += .6

        pose.r = gymapi.Quat(grasp_quat[0], grasp_quat[1],
                             grasp_quat[2], grasp_quat[3])

        gripper = GripperOnly(self.gym, self.sim, self.env, self.isaac_base, self.curr_env_number, pose)
        self.initial_Hgrip = Transform_2_H(pose)
        return gripper

    def get_state(self, rb_states=None):
        if rb_states is None:
            _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
            rb_states = gymtorch.wrap_tensor(_rb_states)

        rb_state = rb_states[0]
        dof_pos  = rb_states[1]
        dof_vel  = rb_states[2]

        ## get object state
        obj_state = self.obj.get_state(rb_state)

        hand_state = self.gripper.get_state(rb_state, dof_pos = dof_pos, dof_vel = dof_vel)

        return {**hand_state, **obj_state}

    def step(self, a=None):
        self.gripper.set_action(a)

    def reset(self, state_dict={}):
        if 'obj_state' in state_dict:
            self.obj.reset(state_dict['obj_state'])
        else:
            self.obj.reset(self.initial_obj_pose)

        if 'grip_state' in state_dict:
            self.gripper.reset(state_dict['grip_state'])
        else:
            self.gripper.reset(self.initial_Hgrip)
