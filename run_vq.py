import os
from datetime import datetime
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from visualization.play_helper import play_policy
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args
from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper
from isaacgym import gymutil    
from OnlineAdaptation.configs.vq_training_config import EnvCfg,RunnerCfg
from OnlineAdaptation.runners.vq_onpolicy_runner import Runner
from OnlineAdaptation.modules.ac import VQActorCritic
import torch 
import argparse 
def get_train_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "littledog_CPG", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--play", "action": "store_true", "default": False, "help": "Play learned policy and record frames"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},

        #! For VQ 
        {"name": "--n_heads", "type": int,"default": -1 },
        {"name": "--codebook_size", "type": int,"default": -1},
        {"name": "--use_forward","type": int,"default": -1},
        {"name": "--stop_gradient", "type": int,"default": -1},

        {"name": "--model_name",'type':str, "default": "VQ"},
        
    ]
    # parse arguments
    parser = argparse.ArgumentParser(description="Training")
    # args = gymutil.parse_arguments(
    #     description="RL Policy",
    #     custom_parameters=custom_parameters)

    args  =gymutil.parse_custom_arguments(
        parser=parser,
        custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def launch(args):
    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()
    _,train_cfg = update_cfg_from_args(None,train_cfg,args)
    
    #! modify args 
    if args.n_heads != -1:
        train_cfg.policy.n_heads = args.n_heads
    if args.use_forward != -1:
        train_cfg.algorithm.use_forward = True if args.use_forward == 1 else False
    if args.stop_gradient != -1:
        train_cfg.algorithm.stop_gradient = True if args.stop_gradient == 1 else False
    if args.codebook_size != -1:
        train_cfg.policy.codebook_size = args.codebook_size
    
    train_cfg.runner.run_name = args.model_name

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=args.headless, 
                                    cfg = env_cfg,eval_cfg=None)
    env = HistoryWrapper(env) 
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    
    
    train_cfg_dict = class_to_dict(train_cfg)
    runner = Runner(env,log_dir,train_cfg,device=args.rl_device)
    return env, runner ,env_cfg ,train_cfg

def play(arg, path = None):
    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    # load policy
    headless = False
    
    device = args.rl_device
    env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=headless, 
                                    cfg = env_cfg,eval_cfg=None)
    env = HistoryWrapper(env) 
    policy_cfg  = class_to_dict(train_cfg.policy)
    policy = VQActorCritic(env.num_obs,env.num_privileged_obs,env.num_obs_history,
                              env.num_history,
                              env.num_actions,
                                        **policy_cfg).to(device)
    
    env.set_apply_force(0, 50, z_force_norm = 0)
    if path is not None:
        policy.load_state_dict(torch.load(path)['model_state_dict'])
    play_policy(env_cfg,train_cfg,policy,env,cmd_vel = [1.,0.0,0.0],
                move_camera=False,record=True)

if __name__ == '__main__':
    args = get_train_args()
    env, runner , env_cfg ,train_cfg = launch(args)
    runner.learn(num_learning_iterations=10000)
    exit()

    
    
