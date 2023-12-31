from legged_gym.envs.configs.basic_config import BasicCfg,BasicRunnerCfg
import numpy as np 


class EnvCfg(BasicCfg):
    class env(BasicCfg.env):
        num_envs = 4096
        num_observations = 52 # 12(joint_pos) + 12(joint_vel) + 12 + 4 + 3 + 3 + 3 
        num_actions = 12
        num_observation_history = 5
        episode_length_s = 20  # episode length in seconds

        # ----------- Basic Observation ------------
        ## 所有 vel 都看
        observe_vel = True
        observe_contact_states = True 
        observe_command = True 

        ## 手动 gait 设定的 task, 必须 observe gait command, clock 和 giat indice 才有意义
        observe_two_prev_actions = False
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_imu = False

        # ---------- Privileged Observations ----------
        num_privileged_obs = 3 + 4 
        privileged_future_horizon = 0
        priv_observe_friction = True #! 1
        priv_observe_restitution = True #! 1 
        priv_observe_base_mass = True #! 1
        priv_observe_com_displacement = False #! 3
        priv_observe_motor_strength = False #! 12
        priv_observe_force_apply = True #! 1 + 3, body_index, Force
        
    class terrain(BasicCfg.terrain):   
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # Height Map only:
        measure_heights = False #! 17 * 11
        measured_points_x = np.linspace(-0.8,0.8,17).tolist()
        measured_points_y = np.linspace(-0.5,0.5,11).tolist()
        # Footclearance only:
        measure_foot_clearance = False #! 4 
        # Footheight only:
        foot_offset = 0.02
        measure_foot_heights = False #! 3 * 3 * 4 
        measured_foot_points_x = [-0.1,0.0,0.1]
        measured_foot_points_y = [-0.1,0.0,0.1]
        
        
        curriculum = True # True
        selected = True # False # select a unique terrain type and pass all arguments
        
        terrain_kwargs = {
            "plane_terrain":{
                "weight": 10.0,
                "height" : 0.0
            },
            'random_uniform_terrain': {
                "weight": 1.0,
                "min_height" : -0.03,
                "max_height" : 0.03,
                "step" : 0.005,
                "downsampled_scale" : 0.2
            },
            # 'sloped_terrain': {
            #     "weight": 1.0,
            #     "slope" : 0.4
            # },
            'pyramid_sloped_terrain': {
                "weight": 1.0,
                "slope" : 0.4,
                'platform_size':3.0
            },
            'wave_terrain':{
                "weight": 1.0,
                'num_waves': 2,
                'amplitude': 0.5,
            },
            # 'stairs_terrain':{
            #     "weight": 1.0,
            #     'step_height': 0.15,
            #     'step_width': 0.5,
            # },
            'pyramid_stairs_terrain':{
                "weight": 1.0,
                'step_width': 0.5,
                'step_height': 0.15,
                'platform_size': 2.0,
            },
            'stepping_stones_terrain':{
                "weight": 1.0,
                'stone_size': 1.5,
                'stone_distance': 0.1,
                'max_height': 0.0,
                'platform_size': 4.,
                'depth': -10
            },
            'discrete_obstacles_terrain':{
                "weight": 1.0,
                "max_height":0.2, 
                "min_size" :1.0, 
                "max_size" : 2.0, 
                "num_rects" : 20, 
                "platform_size":1.
            },
            
        }
        

        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)


    class commands(BasicCfg.commands):

        command_curriculum = True
        cmd_cfg = {
            0:{
                'name':'vel_x',
                'init_low':-0.5,
                'init_high':0.5,
                'limit_low':-1.0,
                'limit_high':1.0,
                'local_range':0.5,
                'num_bins':5,
            },
            1:{
                'name':'vel_y',
                'init_low':-0.3,
                'init_high':0.3,
                'limit_low':-0.3,
                'limit_high':0.3,
                'local_range':0.5,
                'num_bins':3,
            },
            2:{
                'name':'vel_yaw',
                'init_low':-0.2,
                'init_high':0.2,
                'limit_low':-0.3,
                'limit_high':0.3,
                'local_range':0.5,
                'num_bins':3,
            }

        }
        
        num_commands = 3
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y =  [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
        
    class rewards(BasicCfg.rewards):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = False

        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.9
        soft_dof_pos_limit = 0.9 #! 阻止出现在 joint limit 的情况
        base_height_target = 0.25
        max_contact_force = 100. 

        sigma_rew_neg = 0.02

    class reward_scales:
        torques = -0.0002  # -0.0002
        dof_pos_limits = -10.0
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0 # -2.0
        ang_vel_xy = -0.05 # -0.05
        orientation = -0.
        dof_vel = -0.
        dof_acc = -5e-7 # -2.5e-7
        collision = -1. # -1.0
        body_height_v2 = -1.0
        action_rate = -0.01#
        hip_rotate = -0.01 

    class domain_rand(BasicCfg.domain_rand):
        rand_interval_s = 10
        randomize_rigids_after_start = False
        randomize_friction = False
        friction_range = [0.3, 1.25] # increase range
        randomize_restitution = False
        restitution_range = [0.0, 0.4]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        randomize_com_displacement = False
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_lag_timesteps = False
        lag_timesteps = 6
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class force_apply:
        apply_force = True 
        push_interval = 200
        resampling_time = 10.
        # body_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1] # 0-17, 18 bodies
        # body_index = [0,2,3,4,6,7,8,10,11,12,14,15,16,-1] # 0-17, 18 bodies
        body_index = [-1,0, 2,6,10,14, 3,7,11,15] # 先只做 base, thigh, calf 的 force apply, 不做 hip 和 foot
        max_force = 50.0
        min_force = 10.0
        max_z_force = 10.0 
    
    class asset(BasicCfg.asset):
        terminate_after_contacts_on = ["base","hip","thigh",]

