import numpy as np 
from legged_gym.utils.helpers import class_to_dict,parse_sim_params,update_cfg_from_args,get_args
from legged_gym.envs.Go1.legged_robot import LeggedRobot
import copy 
import torch


def load_pri_policy(model_path,env,policy_cfg):
    from privilege.modules.ac import ActorCritic
    actor_critic = ActorCritic(env.num_obs,
                                env.num_privileged_obs,
                                env.num_obs_history,
                                env.num_actions,
                                      **policy_cfg,
                                      ).to(env.device)
    actor_critic.load_state_dict(torch.load(model_path)['model_state_dict'])
    def policy(obs, info={}):
        action = actor_critic.act_inference(obs)
        return action
    return policy

def play_pri(args,headless=True):
    from legged_gym.envs.configs.prvilidge_config import EnvCfg,RunnerCfg,PlayEnvCfg
    from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper
    
    env_cfg = PlayEnvCfg()
    runner_cfg = RunnerCfg()
    policy_cfg = class_to_dict(runner_cfg.policy)
    env_cfg.env.num_envs = 10 
    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=headless, 
                                    cfg = env_cfg,eval_cfg=None)
    env = HistoryWrapper(env)
    model_path = "model_archives/model_20000.pt"
    # policy = load_policy(model_path,env,policy_cfg)
    policy_cfg = class_to_dict(runner_cfg.policy)
    policy = load_pri_policy(model_path,env,policy_cfg)
    num_eval_steps = 2500

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.5, 0.0, 0.0
    obs = env.reset()

    for i in range(num_eval_steps):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        obs, rew, done, info = env.step(actions)


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    args = get_args()
    play_pri(args,headless=False)