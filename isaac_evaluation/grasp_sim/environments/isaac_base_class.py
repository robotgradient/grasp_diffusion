from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
import math
import time
import os, os.path as osp
import copy
from scipy.spatial.transform import Rotation as R
import numpy as np

PHYSICS = 'PHYSX'


class IsaacGymWrapper():

    def __init__(self, env, viewer=True, physics=PHYSICS, freq = 250, device = 'cuda:0',
                 num_spaces = 1, env_args=None, z_convention=False):

        self.franka_urdf = None

        ## Args
        self.sim_device_type, self.compute_device_id = gymutil.parse_device_str(device)
        self.device = device
        self.physics = physics
        self.freq = freq
        self.num_spaces = num_spaces
        self._set_transforms()
        self.z_convention = z_convention

        self.visualize = viewer

        ## Init Gym and Sim
        self.gym = gymapi.acquire_gym()
        self.sim, self.sim_params = self._create_sim()

        ## Create Visualizer
        if (self.visualize):
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
        else:
            self.viewer = None

        ## Create Environment
        self._create_envs(env, env_args)

        ## Update camera pose
        if self.visualize:
            self._reset_camera(env_args)

    def _create_sim(self):
        """Set sim parameters and create a Sim object."""
        # Set simulation parameters

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.freq
        sim_params.gravity = gymapi.Vec3(0, -9.81, 0)
        sim_params.substeps = 1

        # Set stress visualization parameters
        sim_params.stress_visualization = True
        sim_params.stress_visualization_min = 1.0e2
        sim_params.stress_visualization_max = 1e5

        if self.z_convention:
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0., 0., -9.81)

        if self.physics == 'FLEX':
            sim_type = gymapi.SIM_FLEX
            print('using flex engine...')

            # Set FleX-specific parameters
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 10
            sim_params.flex.num_inner_iterations = 200
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8

            sim_params.flex.deterministic_mode = True

            # Set contact parameters
            sim_params.flex.shape_collision_distance = 5e-4
            sim_params.flex.contact_regularization = 1.0e-6
            sim_params.flex.shape_collision_margin = 1.0e-4
            sim_params.flex.dynamic_friction = 0.7
        else:
            sim_type = gymapi.SIM_PHYSX
            print("using physx engine")
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 25
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = 2
            sim_params.physx.use_gpu = True

            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005

        # Create Sim object
        gpu_physics = self.compute_device_id
        if self.visualize:
            gpu_render = 0
        else:
            gpu_render = -1

        return self.gym.create_sim(gpu_physics, gpu_render, sim_type,
                                   sim_params), sim_params

    def _create_envs(self, env, env_args):
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        if self.z_convention:
            plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Set up the env grid - only 1 object for now
        num_envs = self.num_spaces
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Some common handles for later use
        self.envs_gym = []
        self.envs = []
        print("Creating %d environments" % num_envs)
        num_per_row = int(math.sqrt(num_envs))

        for i in range(num_envs):
            if isinstance(env_args, list):
                env_arg = env_args[i]
            else:
                env_arg = env_args

            # create env
            env_i = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs_gym.append(env_i)
            # now initialize the respective env:
            self.envs.append(env(self.gym, self.sim, env_i, self, i, env_arg))

    def _set_transforms(self):
        """Define transforms to convert between Trimesh and Isaac Gym conventions."""
        self.from_trimesh_transform = gymapi.Transform()
        self.from_trimesh_transform.r = gymapi.Quat(0, 0.7071068, 0,
                                                    0.7071068)
        self.neg_rot_x_transform = gymapi.Transform()
        self.neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
        self.neg_rot_x_transform.r = self.neg_rot_x

    def _reset_camera(self, args):
        if self.z_convention is False:
            # Point camera at environments
            cam_pos = gymapi.Vec3(0.0, 1.0, 0.6)
            cam_target = gymapi.Vec3(0.0, 0.8, 0.2)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        else:
            # Point camera at environments
            if args is not None:
                if 'cam_pose' in args:
                    cam = args['cam_pose']
                    cam_pos = gymapi.Vec3(cam[0], cam[1], cam[2])
                    cam_target = gymapi.Vec3(cam[3], cam[4], cam[5])
                else:
                    cam_pos = gymapi.Vec3(0.0, 0.9, 1.3)
                    cam_target = gymapi.Vec3(0.0, 0.0, .7)
            else:
                cam_pos = gymapi.Vec3(0.0, 0.9, 1.3)
                cam_target = gymapi.Vec3(0.0, 0.0, .7)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def get_franka_rpy(self, trimesh_grasp_quat):
        """Return RPY angles for Panda joints based on the grasp pose in the Z-up convention."""
        neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
        rot_z = gymapi.Quat(0, 0, 0.7071068, 0.7071068)
        desired_transform = neg_rot_x * trimesh_grasp_quat * rot_z
        r = R.from_quat([
            desired_transform.x, desired_transform.y, desired_transform.z,
            desired_transform.w
        ])
        return desired_transform, r.as_euler('ZYX')

    def reset(self, state=None):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment.
        '''
        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset(state[idx])
            else:
                env_i.reset()

        return self._evolve_step()

    def reset_robot(self, state=None, ensure_gripper_reset=False):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment. This function only resets the robot
        '''
        # if gripper reset should be ensured, we require two timesteps:
        if (ensure_gripper_reset):
            for idx, env_i in enumerate(self.envs):
                if state is not None:
                    env_i.reset_robot(state[idx],zero_grip_torque=True)
                else:
                    env_i.reset_robot(zero_grip_torque=True)
            self._evolve_step()


        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset_robot(state[idx])
            else:
                env_i.reset_robot()

        return self._evolve_step()

    def reset_obj(self, state=None):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment. This function only resets the robot
        '''
        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset_obj(state[idx])
            else:
                env_i.reset_obj()

        return self._evolve_step()

    def get_state(self):

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)
        rb_states = rb_states.view(self.num_spaces, -1, rb_states.shape[-1])

        # DOF state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        n_dofs = self.envs[0].n_dofs
        dof_vel = dof_states[:, 1].view(self.num_spaces, n_dofs, 1)
        dof_pos = dof_states[:, 0].view(self.num_spaces, n_dofs, 1)

        s = []
        for idx, env_i in enumerate(self.envs):
            s.append(env_i.get_state([rb_states[idx,...], dof_pos[idx, ...], dof_vel[idx, ...]]))

        return s

    def step(self, action=None):

        if action is not None:
            for idx, env_i in enumerate(self.envs):
                env_i.step(action[idx])

        return self._evolve_step()

    def _evolve_step(self):
        # get the sim time
        t = self.gym.get_sim_time(self.sim)

        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Step rendering
        self.gym.step_graphics(self.sim)

        if self.visualize:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

        return self.get_state()

    def kill(self):
        self.gym.destroy_viewer(self.viewer)
        for env in self.envs_gym:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)




