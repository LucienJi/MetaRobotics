import os
from datetime import datetime
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from visualization.play_helper import play_policy
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper

from OnlineAdaptation.configs.training_config import EnvCfg,RunnerCfg
from OnlineAdaptation.runners.onpolicy_runner import Runner
from OnlineAdaptation.modules.ac import GraphActorCritic
import torch 


def launch(args):
    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()

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
    
    _,train_cfg = update_cfg_from_args(None,train_cfg,args)
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
    policy = GraphActorCritic(env.num_obs,env.num_privileged_obs,env.num_obs_history,
                              env.num_history,
                              env.num_actions,
                                        **policy_cfg).to(device)
    
    env.set_apply_force(0, 50, z_force_norm = 0)
    if path is not None:
        policy.load_state_dict(torch.load(path)['model_state_dict'])
    play_policy(env_cfg,train_cfg,policy,env,cmd_vel = [1.,0.0,0.0],
                move_camera=False,record=True)

if __name__ == '__main__':
    args = get_args()
    if args.play:
        path = "logs/Debug/Nov13_09-09-03_Guide/model_5000.pt"
        play(args,path)
        exit()
    else:
        env, runner , env_cfg ,train_cfg = launch(args)
        runner.learn(num_learning_iterations=10000)

    
    
