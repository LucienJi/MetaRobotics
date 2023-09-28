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
def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower



# parse arguments
args = gymutil.parse_arguments()
# initialize gym
gym = gymapi.acquire_gym()
device = args.sim_device

def compute_torques(delta_joint_pos,default_dof_pos,
                    cur_dof_pos,cur_dof_vel):
    if not isinstance(delta_joint_pos, torch.Tensor):
        delta_joint_pos = to_torch(delta_joint_pos, device=device)
    cur_dof_pos = to_torch(cur_dof_pos, device=device)
    cur_dof_vel = to_torch(cur_dof_vel, device=device)
    joint_pos_target = delta_joint_pos + default_dof_pos
    torques = 20.0 * (joint_pos_target - cur_dof_pos )- \
              0.5 * cur_dof_vel
    return torques


def create_sim(args):
    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_FLEX:
        print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
        args.physics_engine = gymapi.SIM_PHYSX
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()
    return sim 


def create_terrain(env_cfg,sim):
    
    terrain = Customized_Terrain(env_cfg.terrain)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]
    tm_params.transform.p.x = -terrain.cfg.border_size
    tm_params.transform.p.y = -terrain.cfg.border_size
    tm_params.transform.p.z = 0.0
    gym.add_triangle_mesh(sim, 
                        terrain.vertices.flatten(order='C'), 
                        terrain.triangles.flatten(order='C'), 
                        tm_params)
    height_samples = torch.tensor(terrain.heightsamples, dtype=torch.float32).view(terrain.tot_rows,
                                                                                   terrain.tot_cols,).to(device)
    env_origins = torch.from_numpy(terrain.env_origins).to(device).to(torch.float)
    return height_samples,env_origins

# create envs 
def create_envs(env_cfg:EnvCfg,sim,n_envs):
    global default_dof_pos
    asset_path = env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = env_cfg.asset.default_dof_drive_mode
    asset_options.collapse_fixed_joints = env_cfg.asset.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = env_cfg.asset.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = env_cfg.asset.flip_visual_attachments
    asset_options.fix_base_link = env_cfg.asset.fix_base_link
    asset_options.density = env_cfg.asset.density
    asset_options.angular_damping = env_cfg.asset.angular_damping
    asset_options.linear_damping = env_cfg.asset.linear_damping
    asset_options.max_angular_velocity = env_cfg.asset.max_angular_velocity
    asset_options.max_linear_velocity = env_cfg.asset.max_linear_velocity
    asset_options.armature = env_cfg.asset.armature
    asset_options.thickness = env_cfg.asset.thickness
    asset_options.disable_gravity = env_cfg.asset.disable_gravity
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    dof_names = gym.get_asset_dof_names(robot_asset)
    dof_props_asset = gym.get_asset_dof_properties(robot_asset)
    rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)

    
    base_init_state_list = env_cfg.init_state.pos + env_cfg.init_state.rot + env_cfg.init_state.lin_vel + env_cfg.init_state.ang_vel
    base_init_state = to_torch(base_init_state_list, device=device, requires_grad=False)
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(*base_init_state[:3])
    env_lower = gymapi.Vec3(0., 0., 0.)
    env_upper = gymapi.Vec3(0., 0., 0.)
    env_origins = torch.zeros(n_envs, 3, device=device, requires_grad=False)
    env_origins[torch.arange(n_envs, device=device)] = terrain_origins[0,0] # 先不管位置
    actor_handles = []
    envs = [] 
    for i in range(n_envs):
        env_handle = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(n_envs)))
        pos = env_origins[i].clone()
        pos[:2] += torch_rand_float(-1., 1., (2,1), device=device).squeeze(1)
        start_pose.p = gymapi.Vec3(*pos)
        print(start_pose.p)

        rigid_shape_props = rigid_shape_props_asset
        gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
        handle = gym.create_actor(env_handle, robot_asset, start_pose, "go1", i,
                                                  env_cfg.asset.self_collisions, 0)
        dof_props = dof_props_asset
        gym.set_actor_dof_properties(env_handle, handle, dof_props)
        body_props = gym.get_actor_rigid_body_properties(env_handle, handle)
        gym.set_actor_rigid_body_properties(env_handle, handle, body_props, recomputeInertia=True)
        envs.append(env_handle)
        actor_handles.append(handle)
    for i in range(12):
        name = dof_names[i]
        angle = env_cfg.init_state.default_joint_angles[name]
        default_dof_pos[i] = angle
    default_dof_pos.unsqueeze(0)
    return envs, actor_handles

# create viewer
def create_viewer(sim):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    # subscribe to spacebar event for reset
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "w")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "a")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "s")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "d")
    return viewer

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

def refresh_gym():
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.render_all_camera_sensors(sim)
def step(desired_dof_pos):
    global default_dof_pos,dof_pos,dof_vel
    for _ in range(5):
        torques = compute_torques(desired_dof_pos,default_dof_pos,dof_pos,dof_vel).cpu()
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)
        gym.refresh_dof_state_tensor(sim)
    refresh_gym()
env_cfg = EnvCfg() 
sim = create_sim(args)

height_samples,terrain_origins = create_terrain(env_cfg,sim)

camera_pos = np.array([3.0,-2.0, 1.0])
theta = 0.0
camera_direction = np.array([np.cos(theta), np.sin(theta), 0.])  

default_dof_pos = torch.zeros(12, dtype=torch.float, device=device, requires_grad=False)
n_envs = 1
num_dof = 12 
envs,actor_handles = create_envs(env_cfg,sim,n_envs)

viewer = create_viewer(sim)

set_camera(viewer,camera_pos, camera_direction)


actor_root_state = gym.acquire_actor_root_state_tensor(sim)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)

dof_state = gymtorch.wrap_tensor(dof_state_tensor) # base frame
root_states = gymtorch.wrap_tensor(actor_root_state) # world frame
dof_pos = dof_state.view(n_envs, num_dof, 2)[..., 0]
dof_vel = dof_state.view(n_envs, num_dof, 2)[..., 1]

while not gym.query_viewer_has_closed(viewer):

    handle_viewer(viewer)
    desired_dof_pos = np.zeros(shape=(n_envs, num_dof), dtype=np.float32)

    step(desired_dof_pos)
    # step the physics
    # gym.simulate(sim)
    # gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
