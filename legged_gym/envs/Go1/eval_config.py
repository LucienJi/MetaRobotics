from .mo_quad_config import BasicCfg, BasicRunnerCfg, MoQuadCfg,MoQuadRunnerCfg
class EvalCfg(BasicCfg):
    class env(BasicCfg.env):
        num_envs = 50
        # num_observation_history = 1
    class terrain(BasicCfg.terrain):
        # mesh_type = 'trimesh'
        mesh_type = 'plane'
    class domain_rand(BasicCfg.domain_rand):
        randomize_lag_timesteps = False
        randomize_base_mass = False
        randomize_friction = False
        randomize_restitution = False
        randomize_motor_strength = False
        randomize_gravity = False
        randomize_motor_offset = False
    class noise(BasicCfg.noise):
        add_noise = False
    
    class reward_scales(MoQuadCfg.reward_scales):
        pass 
    class reward_ranges(MoQuadCfg.reward_ranges):
        pass
class EvalRunnerCfg(MoQuadRunnerCfg):
    class algorithm(MoQuadRunnerCfg.algorithm):
        use_control = True 
    class runner(MoQuadRunnerCfg.runner):
        run_name = 'Test'
        experiment_name = 'Eval'