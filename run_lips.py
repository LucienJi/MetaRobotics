import os
from datetime import datetime
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper

from Lips.configs.training_config import EnvCfg,RunnerCfg
from Lips.runners.onpolicy_runner import Runner

def launch(args):
    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    headless = True
    env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=headless, 
                                    cfg = env_cfg,eval_cfg=None)
    env = HistoryWrapper(env) 
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    
    _,train_cfg = update_cfg_from_args(None,train_cfg,args)
    train_cfg_dict = class_to_dict(train_cfg)
    runner = Runner(env,log_dir,train_cfg,device=args.rl_device)
    return env, runner ,env_cfg ,train_cfg


if __name__ == '__main__':
    args = get_args()
    env, runner , env_cfg ,train_cfg = launch(args)
    runner.learn(num_learning_iterations=10000)

    
    
