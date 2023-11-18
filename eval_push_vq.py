import os
from datetime import datetime
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from visualization.play_helper import play_policy
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper

from OnlineAdaptation.configs.vq_training_config import EnvCfg,RunnerCfg
from OnlineAdaptation.runners.vq_onpolicy_runner import Runner
from OnlineAdaptation.modules.ac import VQActorCritic
import torch 
from legged_gym.envs.wrapper.push_eval_wrapper import PushConfig,EvalWrapper


push_config1 = PushConfig(
    id = 0,
    body_index_list = [0],
    change_interval = -1,
    force_list = [0],
)

def eval(args, path = None):
    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

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
    env_pushed = EvalWrapper(env, env_cfg, cmd_vel=[0.5,0.0,0.0],
                      record=False, move_camera=False,experiment_name='Eval')
    env_pushed.set_eval_config(
        eval_config = [push_config1]
    )
    
    policy_cfg  = class_to_dict(train_cfg.policy)
    policy = VQActorCritic(env.num_obs,env.num_privileged_obs,env.num_obs_history,
                              env.num_history,
                              num_actions=env.num_actions,
                              use_forward=train_cfg.algorithm.use_forward,
                                        **policy_cfg).to(device)
    if path is not None:
        policy.load_state_dict(torch.load(path)['model_state_dict'])
    for i in range(int(env.max_episode_length) + 10):
        obs_dict = env_pushed.obs_dict
        with torch.no_grad():
            actions = policy.act_inference(obs_dict)
            env_pushed.step(actions.detach())
    
    res = env_pushed.get_result()
    # print(res)
    for k,v in res.items():
        print(k, v.shape)

if __name__ == '__main__':
    args = get_args()
    path = "logs/VQ/Nov16_22-54-45_NoForward/model_10000.pt"
    eval(args,path)
    exit()