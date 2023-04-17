import numpy as np
from isaac_evaluation.utils.geometry_utils import pq_to_H
import os, os.path as osp
from isaacgym import gymapi, gymtorch
import copy
from se3dif.datasets import AcronymGraspsDirectory
from se3dif.utils import get_data_src
from isaac_evaluation.utils.generate_obj_urdf import generate_obj_urdf



class SimpleObject():
    '''
    A simple Object Isaac Gym class.
    This class takes care of the objects pose or even of the objects pose rearrangement.
    '''
    def __init__(self, gym, sim, env, isaac_base, env_number, pose, obj_id, obj_name, obj_type='Mug',
                 args = None,
                 collision_group=1, segmentationId=0, linearDamping=0, angularDamping=0, scale=1., disable_gravity=True):

        ##Set arguments
        self.args = self._set_args(args)
        self.disable_gravity = disable_gravity

        ## Set Hyperparameters
        self.gym = gym
        self.sim = sim
        self.env = env
        self.isaac_base = isaac_base
        self.initial_pose = copy.deepcopy(pose)

        ##Set args
        self.obj_type = obj_type
        self.obj_id = obj_id
        self.obj_name = obj_name
        self.linearDamping = linearDamping
        self.angularDamping = angularDamping

        ## Set assets
        obj_assets = self._set_assets()
        self.handle = gym.create_actor(env, obj_assets, pose, obj_name,
                                       group=env_number, filter=collision_group, segmentationId=segmentationId)
        print('Object Handle: {}'.format(self.handle))

        self.gym.set_actor_scale(self.env, self.handle, scale)

    def _set_args(self, args):
        if args is None:
            args ={
                'physics':'PHYSX',
            }
        else:
            args = args
        return args

    def _get_objs_path(self):
        acronym_grasps = AcronymGraspsDirectory(data_type=self.obj_type)
        mesh_rel_path = acronym_grasps.avail_obj[self.obj_id].mesh_fname
        mesh_path_file = os.path.join(get_data_src(), mesh_rel_path)
        res_urdf_path = generate_obj_urdf(mesh_path_file)

        return res_urdf_path

    def _set_assets(self):
        asset_file_object = self._get_objs_path()

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        #asset_options.flip_visual_attachments = False
        asset_options.armature = 0.
        asset_options.thickness = 0.
        asset_options.density = 1000.

        asset_options.linear_damping = self.linearDamping  # Linear damping for rigid bodies
        asset_options.angular_damping = self.angularDamping  # Angular damping for rigid bodies
        asset_options.disable_gravity = self.disable_gravity
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 200000

        obj_asset = self.gym.load_asset(
            self.sim, '', asset_file_object, asset_options)
        return obj_asset

    def get_state(self, rb_states=None):
        if rb_states is None:
            _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
            rb_states = gymtorch.wrap_tensor(_rb_states)

        obj_state = rb_states[self.handle,...]
        obj_pos = obj_state[:3]
        obj_rot = obj_state[3:7]
        obj_vel = obj_state[7:]

        H = pq_to_H(obj_pos, obj_rot)

        return {'obj_pos':obj_pos, 'obj_rot':obj_rot, 'obj_vel': obj_vel, 'H_obj':H}

    def get_rigid_body_state(self):
        # gets state of exactly this rigid body
        return self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)

    def reset(self, H):
        pos = [H.p.x, H.p.y, H.p.z]
        rot = [H.r.x, H.r.y, H.r.z, H.r.w]
        self.set_rigid_body_pos(pos, rot)

    def set_rigid_body_pos(self, pos, ori):
        # sets the position of the ridgid body and the velocity to zero
        obj = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_NONE)
        obj['pose']['p'].fill((pos[0],pos[1],pos[2]))
        obj['pose']['r'].fill((ori[0],ori[1],ori[2],ori[3]))
        obj['vel']['linear'].fill((0,0,0))
        obj['vel']['angular'].fill((0,0,0))
        self.gym.set_actor_rigid_body_states(self.env, self.handle, obj, gymapi.STATE_ALL)

    def set_rigid_body_pos_keep_vel(self, pos, ori):
        # sets the position of the ridgid body and keeps the velocity
        obj = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        obj['pose']['p'].fill((pos[0],pos[1],pos[2]))
        obj['pose']['r'].fill((ori[0],ori[1],ori[2],ori[3]))
        self.gym.set_actor_rigid_body_states(self.env, self.handle, obj, gymapi.STATE_ALL)

    def set_rigid_body_pos_vel(self, pos, ori, vel_lin, vel_ang):
        # sets the position and velocity
        obj = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_NONE)
        obj['pose']['p'].fill((pos[0],pos[1],pos[2]))
        obj['pose']['r'].fill((ori[0],ori[1],ori[2],ori[3]))
        obj['vel']['linear'].fill((vel_lin[0],vel_lin[1],vel_lin[2]))
        obj['vel']['angular'].fill((vel_ang[0],vel_ang[1],vel_ang[2]))
        self.gym.set_actor_rigid_body_states(self.env, self.handle, obj, gymapi.STATE_ALL)

