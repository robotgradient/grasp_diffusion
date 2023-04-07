from isaacgym import *
import os, sys
import numpy as np
from se3dif.datasets import AcronymGraspsDirectory
from isaac_evaluation.grasp_sim.environments import IsaacGymWrapper
from isaac_evaluation.grasp_sim.environments.grip_eval_env import GraspingGymEnv

from se3dif.utils import to_numpy, to_torch
from isaac_evaluation.utils.geometry_utils import pq_to_H, H_2_Transform
import torch

class GraspSuccessEvaluator():

    def __init__(self, obj_class, n_envs = 10, idxs=None, viewer=True, device='cpu', rotations=None, enable_rel_trafo=True):
        self.device = device
        self.obj_class = obj_class
        self.grasps_directory = AcronymGraspsDirectory(data_type=obj_class)
        self.n_envs = n_envs
        self.rotations = rotations
        # This argument tells us if the grasp poses are relative w.r.t. the current object pose or not
        self.enable_rel_trafo = enable_rel_trafo

        ## Build Envs ##
        if idxs is None:
            idxs = [0]*n_envs

        grasps = [self.grasps_directory.avail_obj[idx_i] for idx_i in idxs]

        scales = [grasp.mesh_scale for grasp in grasps]
        obj_ids = [idx_i for idx_i in idxs]
        obj_types = [grasp.mesh_type for grasp in grasps]
        # Note: quaternion here already has convention, w,x,y,z
        if not(rotations is None):
            rotations_flat = [rotation for rotation in rotations]
        else:
            rotations_flat = [[0,0,0,1] for idx_i in idxs]

        env_args = self._get_args(obj_ids, obj_types, scales, rotations_flat)


        self.grasping_env = IsaacGymWrapper(env=GraspingGymEnv, env_args=env_args,
                                            z_convention=True, num_spaces=self.n_envs,
                                            viewer = viewer, device=self.device)

        self.success_cases = 0

    def reset(self):
        self.success_cases = 0

    def _get_args(self, obj_ids, obj_types='Mug', scales=1.,rotations=None):
        args = []
        for obj_id, obj_type, scale, rotation in zip(obj_ids, obj_types, scales,rotations):
            obj_args = {
                'object_type': obj_type,
                'object_id': obj_id,
                'object_name': 'mug_01',
                'scale': scale,
                'obj_or': np.asarray(rotation)
            }
            arg = {'obj_args': obj_args}
            args.append(arg)
        return args

    def eval_set_of_grasps(self, H):
        n_grasps = H.shape[0]

        for i in range(0, n_grasps, self.n_envs):
            print('iteration: {}'.format(i))

            batch_H = H[i:i+self.n_envs,...]
            self.eval_batch(batch_H)

        return self.success_cases

    def eval_batch(self, H):

        s = self.grasping_env.reset()
        for t in range(10):
            self.grasping_env.step()
        s = self.grasping_env.reset()
        for t in range(10):
            self.grasping_env.step()

        # 1. Set Evaluation Grasp
        H_obj = torch.zeros_like(H)
        for i, s_i in enumerate(s):
            H_obj[i,...] = pq_to_H(p=s_i['obj_pos'], q=s_i['obj_rot'])
            if not(self.enable_rel_trafo):
                H_obj[i, :3, :3] = torch.eye(3)
        Hg = torch.einsum('bmn,bnk->bmk', H_obj, H)

        state_dicts = []
        for i in range(Hg.shape[0]):
            state_dict = {
                'grip_state': Hg[i,...]
            }
            state_dicts.append(state_dict)

        s = self.grasping_env.reset(state_dicts)

        # 2. Grasp
        policy = GraspController(self.grasping_env, n_envs=self.n_envs)

        T = 700
        for t in range(T):
            a = policy.control(s)
            s = self.grasping_env.step(a)

        self._compute_success(s)
        del policy
        torch.cuda.empty_cache()

    def _compute_success(self, s):
        for si in s:
            hand_pos = si['hand_pos']
            obj_pos  = si['obj_pos']
            ## Check How close they are ##
            distance = (hand_pos - obj_pos).pow(2).sum(-1).pow(.5)

            if distance <0.3:
                self.success_cases +=1


class GraspController():
    '''
     A controller to evaluate the grasping
    '''
    def __init__(self, env, hand_cntrl_type='position', finger_cntrl_type='torque', n_envs = 0):
        self.env = env

        ## Controller Type
        self.hand_cntrl_type = hand_cntrl_type
        self.finger_cntrl_type = finger_cntrl_type

        self.squeeze_force = .6
        self.hold_force = [self.squeeze_force]*n_envs

        self.r_finger_target = [0.]*n_envs
        self.l_finger_target = [0.]*n_envs
        self.grasp_count = [0]*n_envs

        ## State Machine States
        self.control_states = ['approach', 'grasp', 'lift']
        self.grasp_states = ['squeeze', 'hold']

        self.state = ['grasp']*n_envs
        self.grasp_state = ['squeeze']*n_envs

    def set_H_target(self, H):
        self.H_target = H
        self.T = H_2_Transform(H)

    def control(self, states):
        actions = []
        for idx, state in enumerate(states):
            if self.state[idx] =='approach':
                action = self._approach(state, idx)
            elif self.state[idx] == 'grasp':
                if self.grasp_state[idx] == 'squeeze':
                    action = self._squeeze(state, idx)
                elif self.grasp_state[idx] == 'hold':
                    action = self._hold(state, idx)
            elif self.state[idx] == 'lift':
                action = self._lift(state, idx)

            actions.append(action)
        return actions

    def _approach(self, state, idx=0):
        hand_pose  = state[0]['hand_pos']
        hand_rot  = state[0]['hand_rot']

        target_pose = torch.Tensor([self.T.p.x, self.T.p.y, self.T.p.z])

        pose_error = hand_pose - target_pose
        des_pose = hand_pose - pose_error*0.1

        ## Desired Pos for left and right finger is 0.04
        r_error = state[0]['r_finger_pos'] - .04
        l_error = state[0]['l_finger_pos'] - .04

        K = 1000
        D = 20
        ## PD Control law for finger torque control
        des_finger_torque = torch.zeros(2)
        des_finger_torque[1:] += -K*r_error - D*state[0]['r_finger_vel']
        des_finger_torque[:1] += -K*l_error - D*state[0]['l_finger_vel']

        action = {'hand_control_type': 'position',
                  'des_hand_position': des_pose,
                  'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque
                  }

        ## Set State Machine Transition
        error = pose_error.pow(2).sum(-1).pow(.5)
        if error<0.005:
            print('start grasp')
            self.state[idx] = 'grasp'
        return action

    def _squeeze(self, state, idx=0):
        ## Squeezing should achieve stable grasping / contact with the object of interest
        des_finger_torque = torch.ones(2)*-self.squeeze_force

        action = {'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque}

        ## Set State Machine Transition after an empirical number of steps
        self.grasp_count[idx] +=1
        if self.grasp_count[idx]>300:
            self.grasp_count[idx] = 0
            self.grasp_state[idx] = 'hold'

        return action

    def _hold(self, state, idx=0):

        if self.grasp_count[idx] == 0:
            self.hold_force[idx] = self.squeeze_force
        else:
            self.hold_force[idx] +=1.0
        self.grasp_count[idx] += 1

        ## Set torques
        #print(self.hold_force[idx])
        des_finger_torque = torch.ones(2) * -self.hold_force[idx]
        action = {'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque}

        ## Set State Machine Transition after an empirical number of steps, this also corresponded
        ## to increasing the desired grasping force for 100 steps
        if self.grasp_count[idx] > 100:
            #print(self.hold_force[idx])

            self.grasp_count[idx] = 0.
            self.l_finger_target[idx] = state['l_finger_pos'].clone()
            self.r_finger_target[idx] = state['r_finger_pos'].clone()

            self.state[idx] = 'lift'

        return action

    def _lift(self, state, idx=0):
        obj_pose = state['obj_pos']
        hand_pose  = state['hand_pos']
        #print('hand y: {}, obj y: {}'.format(hand_pose[1], obj_pose[1]))

        target_pose = torch.zeros_like(obj_pose)
        target_pose[2] = 2.
        target_pose[0] = 0.

        ## Set Desired Hand Pose
        pose_error = hand_pose - target_pose
        des_pose = hand_pose - pose_error*0.05

        ## Set Desired Grip Force
        des_finger_torque = torch.ones(2) * -self.hold_force[idx]

        r_error = state['r_finger_pos'] - self.r_finger_target[idx]
        l_error = state['l_finger_pos'] - self.l_finger_target[idx]

        K = 1000
        D = 20
        des_finger_torque[1:] += -K*r_error - D*state['r_finger_vel']
        des_finger_torque[:1] += -K*l_error - D*state['l_finger_vel']


        action = {'hand_control_type': 'position',
                  'des_hand_position': des_pose,
                  'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque
                  }

        return action
