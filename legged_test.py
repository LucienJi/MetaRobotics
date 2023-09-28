
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import quat_apply
import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym import gymutil, gymapi,gymtorch
from isaacgym.terrain_utils import *
from math import sqrt
import math 
from legged_gym.utils.terrain_v2 import Customized_Terrain
from RMA.configs.training_config import EnvCfg
import torch 


args = get_args()
env_cfg = EnvCfg()

env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
sim_params = {"sim":class_to_dict(env_cfg.sim)}
sim_params = parse_sim_params(args, sim_params)

headless = True
env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=headless, 
                                    cfg = env_cfg,eval_cfg=None)
gym = env.gym 
sim = env.sim
viewer = env.viewer 
camera_pos = np.array([3.0,-2.0, 2.0])
theta = 0.0
camera_direction = np.array([np.cos(theta), np.sin(theta), 0.])  


def set_camera(viewer,camera_pos, camera_direction):
    cam_pos = gymapi.Vec3(*camera_pos)
    target = camera_pos +camera_direction
    cam_target = gymapi.Vec3(*target)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def handle_viewer(viewer):
    global theta,camera_pos,camera_direction
    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            # gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
            print("Reset")
        if evt.action == 'w' and evt.value > 0:
            camera_pos += 2.0 * camera_direction
            set_camera(viewer,camera_pos, camera_direction)
        if evt.action == 's' and evt.value > 0:
            camera_pos -= 2.0 *  camera_direction
            set_camera(viewer,camera_pos, camera_direction)
        if evt.action == 'a' and evt.value > 0:
            theta -= 0.1
            camera_direction = np.array([np.cos(theta), np.sin(theta), 0.])  
            set_camera(viewer,camera_pos, camera_direction)
        if evt.action == 'd' and evt.value > 0:
            theta += 0.1
            camera_direction = np.array([np.cos(theta), np.sin(theta), 0.])  
            set_camera(viewer,camera_pos, camera_direction)

n_envs = env.num_envs
num_dof = 12 
env.reset()
while not env.gym.query_viewer_has_closed(env.viewer):

    # handle_viewer(env.viewer)
    desired_dof_pos = torch.from_numpy(np.zeros(shape=(n_envs, num_dof), dtype=np.float32)).to(env.device)
    # desired_dof_pos[0,2] = -1.0
    env.step(actions = desired_dof_pos)
    print("Foot in base: ", env.get_foot_position_base_frame())
    # print("Foot Clearance: ", env.measured_foot_clearance)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)