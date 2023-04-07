import torch
from se3dif.utils import directory_utils
from isaacgym import gymapi, gymtorch
import os, os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_evaluation.utils.geometry_utils import H_2_Transform

class GripperOnly():
    def __init__(self, gym, sim, env, isaac_base, env_number, pose, collision_group=0, segmentationId=0):

        ## Hyperparameters
        self.gym = gym
        self.sim = sim
        self.env = env
        self.isaac_base = isaac_base


        ## Controller Args
        self.hand_cntrl_type    = 'position'
        self.grip_cntrl_type = 'torque'

        ## State args
        self.base_pose = pose

        ## Set assets
        self.franka_asset = self.set_assets()

        # add the franka hand
        self.handle = self.gym.create_actor(self.env, self.franka_asset, pose, "franka", group=env_number, filter=collision_group, segmentationId=segmentationId)
        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env, self.handle, "panda_hand")
        self.base_handle = self.gym.find_actor_rigid_body_handle(self.env, self.handle, "world")

        ## Set initial joint configuration ##
        curr_joint_positions = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        curr_joint_positions['pos'][-1] = 0.04
        curr_joint_positions['pos'][-2] = 0.04

        self.gym.set_actor_dof_states(self.env, self.handle,
                                      curr_joint_positions, gymapi.STATE_ALL)

        curr_joint_positions = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)


        self.gym.set_actor_dof_position_targets(self.env, self.handle, curr_joint_positions['pos'])

        ## Set control properties
        self._set_cntrl_properties()
        self.target_pose = curr_joint_positions['pos']
        self.target_torque = torch.zeros(self.target_pose.shape[0])

        # Attractor

        self._set_initial_target = True

    def set_assets(self):
        # Load franka asset
        franka_asset_file = osp.join(directory_utils.get_mesh_src(),
                                     'hand_only/robots/franka_panda_hand.urdf')
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.0
        asset_options.thickness = 0.0

        asset_options.linear_damping = 100.0  # Linear damping for rigid bodies
        asset_options.angular_damping = 100.0  # Angular damping for rigid bodies
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = True

        return self.gym.load_asset(
            self.sim, '', franka_asset_file, asset_options)

    def get_state(self, rb_states, dof_pos, dof_vel):

        ## Base state
        base_state = rb_states[self.base_handle, ...]
        base_pos = base_state[:3]
        base_rot = base_state[3:7]
        base_vel_p = base_state[7:10]
        base_vel_r = base_state[10:]

        self.base_state_dict = {'base_pos': base_pos, 'base_rot': base_rot,
                           'base_vel_p': base_vel_p, 'base_vel_r': base_vel_r}

        ## Hand state
        hand_state = rb_states[self.hand_handle, ...]
        hand_pos = hand_state[:3]
        hand_rot = hand_state[3:7]
        #print('pos: {}, ori:{}'.format(hand_pos, hand_rot))

        hand_vel_p = hand_state[7:10]
        hand_vel_r = hand_state[10:]


        hand_state_dict = {'hand_pos': hand_pos, 'hand_rot': hand_rot,
                           'hand_vel_p': hand_vel_p, 'hand_vel_r': hand_vel_r}

        self.robot_state = [dof_pos, dof_vel]
        ## Fingers state
        finger_state_dict = {'r_finger_pos': dof_pos[-1], 'l_finger_pos': dof_pos[-2],
                             'r_finger_vel': dof_vel[-1], 'l_finger_vel': dof_vel[-2],
                             'position_pos': dof_pos[:6]}


        return {**hand_state_dict, **finger_state_dict}

    def _init_cntrl(self):
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e5
        attractor_properties.damping = 5e3
        attractor_properties.axes = gymapi.AXIS_ALL
        attractor_properties.rigid_handle = self.hand_handle

        self.attractor_handle = self.gym.create_rigid_body_attractor(self.env, attractor_properties)
        hand_pose = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)['pose'][:][-4]

        #print('Target pose: {}'.format(hand_pose))

        self.gym.set_attractor_target(self.env, self.attractor_handle, hand_pose)

    def _set_cntrl_properties(self, grip_cntrl_type='torque'):
        # get joint limits and ranges for Franka
        franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = franka_dof_props['lower']
        franka_upper_limits = franka_dof_props['upper']
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
        franka_num_dofs = len(franka_dof_props)

        # set DOF control properties (except grippers)
        franka_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:].fill(400.0)
        franka_dof_props["damping"][:].fill(100.0)

        if grip_cntrl_type =='torque':
            self.grip_cntrl_type = grip_cntrl_type
            # set DOF control properties for grippers
            franka_dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][-2:].fill(0.0)
            franka_dof_props["damping"][-2:].fill(0.0)
        else:
            self.grip_cntrl_type = 'position'
            franka_dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_POS)
            franka_dof_props["stiffness"][-2:].fill(200.0)
            franka_dof_props["damping"][-2:].fill(40.0)

        # Set DOF control properties
        self.gym.set_actor_dof_properties(self.env, self.handle, franka_dof_props)

    def set_action(self, action):

        ## Set initial attractor's target
        if self._set_initial_target:
            self._set_initial_target = False
            self._init_cntrl()

        if 'hand_control_type' in  action:

            if action['hand_control_type']=='position':
                ## Set desired position displacement
                hand_xyz = action['des_hand_position']

                # The attractor is used to move the hand
                attractor_properties = self.gym.get_attractor_properties(self.env, self.attractor_handle)
                pose = attractor_properties.target
                # print('target pos: ({}, {}, {}), target ori: ({} {} {} {})'.format(pose.p.x, pose.p.y, pose.p.z,
                #                                                                    pose.r.x, pose.r.y, pose.r.z, pose.r.w))
                pose.p.x = hand_xyz[0]
                pose.p.y = hand_xyz[1]
                pose.p.z = hand_xyz[2]
                self.gym.set_attractor_target(self.env, self.attractor_handle, pose)


        if 'grip_control_type' in  action:

            if action['grip_control_type'] != self.grip_cntrl_type:
                self._set_cntrl_properties(grip_cntrl_type=action['grip_control_type'])

            if action['grip_control_type']=='torque':
                torque_grip = action['des_grip_torque']
                self.target_torque[-2:] = torque_grip

        ## Set controllers ##
        self.gym.apply_actor_dof_efforts(self.env, self.handle, self.target_torque)

    def reset(self, H):
        ## Set root
        T = H_2_Transform(H)
        self.gym.set_rigid_transform(self.env, self.handle, T)

        ## Set DOF to zero
        curr_joint_positions = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        #print(curr_joint_positions)
        curr_joint_positions['pos'] = np.zeros_like(curr_joint_positions['pos'])
        curr_joint_positions['vel'] = np.zeros_like(curr_joint_positions['vel'])
        curr_joint_positions['pos'][-1] = 0.04
        curr_joint_positions['pos'][-2] = 0.04

        self.gym.set_actor_dof_states(self.env, self.handle,
                                      curr_joint_positions, gymapi.STATE_ALL)

        self.gym.set_actor_dof_position_targets(self.env, self.handle, curr_joint_positions['pos'])

        if self._set_initial_target:
             self._set_initial_target = False
             self._init_cntrl()
        else:
            hand_pose = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)['pose'][:][-4]
            self.gym.set_attractor_target(self.env, self.attractor_handle, hand_pose)

            self.target_torque = torch.zeros_like(self.target_torque)
            self.gym.apply_actor_dof_efforts(self.env, self.handle, self.target_torque)
