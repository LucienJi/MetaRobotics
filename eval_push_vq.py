import os
from datetime import datetime
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from visualization.play_helper import play_policy
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper

from OnlineAdaptation.configs.vq_training_config import RunnerCfg
from OnlineAdaptation.configs.eval_config import EnvCfg
from OnlineAdaptation.modules.ac import VQActorCritic
import torch 
from legged_gym.envs.wrapper.push_eval_wrapper import PushConfig,EvalWrapper
import numpy as np 

stationary_push_config = PushConfig(
    id = 0,
    body_index_list = [4],
    change_interval = 100,
    force_list = [100],

)


def eval_stationary(args, path = None , cmd_vel =[0.5,0.0,0.0],name='Eval',
                    eval_path = "logs/Eval"):
    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    # load policy
    headless = args.headless
    
    device = args.rl_device
    env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=headless, 
                                    cfg = env_cfg,eval_cfg=None)
    env = HistoryWrapper(env) 
    env_pushed = EvalWrapper(env, env_cfg, cmd_vel=cmd_vel,
                      record=False, move_camera=False,experiment_name='Eval')
    env_pushed.set_eval_config(
        eval_config = [stationary_push_config]
    )
    eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + "-stationary_push-" + name
    policy_cfg  = class_to_dict(train_cfg.policy)
    policy = VQActorCritic(env.num_obs,env.num_privileged_obs,env.num_obs_history,
                              env.num_history,
                              num_actions=env.num_actions,
                              use_forward=train_cfg.algorithm.use_forward,
                                        **policy_cfg).to(device)
    if path is not None:
        policy.load_state_dict(torch.load(path)['model_state_dict'])
    policy.eval()
    for i in range(int(env.max_episode_length) + 10):
        obs_dict = env_pushed.obs_dict
        with torch.no_grad():
            actions = policy.act_inference(obs_dict)
            env_pushed.step(actions.detach(), draw=True)
    
    eval_res = env_pushed.get_result()

    if os.path.exists(eval_path) == False:
        os.makedirs(eval_path)
    eval_file_name = os.path.join(eval_path,eval_name)
    np.save(eval_file_name, eval_res)



if __name__ == '__main__':
    args = get_args()
    path = "logs/VQ/Dec04_22-47-20_STG_4_head/model_10000.pt"
    eval_stationary(args,path,
                    cmd_vel=[0.1,0.0,0.0],
                    name = 'DebugEval',
                    eval_path = "logs/Eval")
    print("Done")
    exit()