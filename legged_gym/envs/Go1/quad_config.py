from .basic_config import BasicCfg, BasicRunnerCfg
class QuadCfg(BasicCfg):
    class env(BasicCfg.env):
        num_envs = 4096
        
    class terrain(BasicCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
    
    #! Customized
    class commands(BasicCfg.commands):
        ang_vel_yaw = [-0.2, 0.2]
        body_height_cmd =  [-0.1, 0.1]
        gait_frequency_cmd_range = [2.0, 4.0]
        footswing_height_range =  [0.1, 0.15]

        ###########
        limit_vel_yaw = [-1.0, 1.0]
        limit_gait_frequency = [1.0, 6.0]
        limit_footswing_height =  [0.03, 0.35]


        ###########
        num_bins_vel_yaw = 11
        num_bins_body_height = 5
        num_bins_gait_frequency = 5
        num_bins_footswing_height = 5
        num_bins_stance_width = 3
        num_bins_stance_length = 3
    
    
    class domain_rand(BasicCfg.domain_rand):
        randomize_rigids_after_start = False
        randomize_base_mass = True
        added_mass_range = [-1, 3]

        push_robots = False
        max_push_vel_xy = 0.5

        randomize_friction = True
        friction_range = [0.1, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        restitution = 0.5  # default terrain restitution
        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        rand_interval_s = 4

    class rewards(BasicCfg.rewards):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02
        base_height_target = 0.34
        max_contact_force = 100. 


    class reward_ranges:
        pass 

class QuadRunnerCfg(BasicRunnerCfg):
    class algorithm(BasicRunnerCfg.algorithm):
        # algorithm
        entropy_coef = 0.01
        # entropy_coef = 0.0
        learning_rate = 1.e-3 # 5.e-4
        adaptation_module_learning_rate = 1.e-3
        
    class runner(BasicRunnerCfg.runner):
        run_name = 'Skill'
        experiment_name = 'Expert'
        
        