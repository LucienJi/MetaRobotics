import os
from datetime import datetime
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from visualization.play_helper import play_policy
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper
#----------------- Load Baselines -----------------
from Expert.configs.push_training_config import RunnerCfg as ExpertRunnerCfg
from Expert.modules.ac import ActorCritic as ExpertActorCritic

from RMA.configs.push_training_config import RunnerCfg as RMARunnerCfg 
from RMA.modules.ac import ActorCritic as RMAActorCritic

from EstimatorNet.configs.push_training_config import RunnerCfg as EstimatorNetRunnerCfg
from EstimatorNet.modules.ac import ActorCritic as EstimatorNetActorCritic

from OnlineAdaptation.configs.eval_config import EnvCfg
import torch 
from legged_gym.envs.wrapper.push_eval_wrapper import PushConfig,EvalWrapper
import numpy as np 

stationary_push_config = PushConfig(
    id = 0,
    body_index_list = [4],
    change_interval = 100,
    force_list = [50],

)


def eval_stationary(args, path = None , cmd_vel =[0.5,0.0,0.0],name='Eval', baseline_name = 'Expert',
                    eval_path = "logs/Eval"):
    env_cfg = EnvCfg()
    if baseline_name == 'expert':
        train_cfg = ExpertRunnerCfg()
        AC = ExpertActorCritic
    elif baseline_name == 'rma':
        train_cfg = RMARunnerCfg()
        AC = RMAActorCritic
    elif baseline_name == 'e_net':
        train_cfg = EstimatorNetRunnerCfg()
        AC = EstimatorNetActorCritic
    else:
        raise NotImplementedError
    
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
    eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + "-stationary_push-" +baseline_name+ "-" + name
    policy_cfg  = class_to_dict(train_cfg.policy)
    policy = AC(env.num_obs,
                env.num_privileged_obs,
                env.num_obs_history,
                num_actions=env.num_actions,
                **policy_cfg).to(device)
    if path is not None:
        policy.load_state_dict(torch.load(path)['model_state_dict'],strict=False)
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

def eval_stationary_multiforce(args, path , cmd_vel =[0.5,0.0,0.0],name='Eval', baseline_name = 'Expert',
                    eval_path = "logs/Eval",
                    force_list = [50,100,150,200,250,300,350,400,450,500]):
    env_cfg = EnvCfg()
    if baseline_name == 'expert':
        train_cfg = ExpertRunnerCfg()
        AC = ExpertActorCritic
    elif baseline_name == 'rma':
        train_cfg = RMARunnerCfg()
        AC = RMAActorCritic
    elif baseline_name == 'e_net':
        train_cfg = EstimatorNetRunnerCfg()
        AC = EstimatorNetActorCritic
    else:
        raise NotImplementedError
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    # load policy
    headless = args.headless

    policy_cfg  = class_to_dict(train_cfg.policy)
    policy = AC(env.num_obs,
                env.num_privileged_obs,
                env.num_obs_history,
                num_actions=env.num_actions,
                **policy_cfg).to(device)
    policy.load_state_dict(torch.load(path)['model_state_dict'],strict=False)
    policy.eval()
    
    device = args.rl_device
    env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=headless, 
                                    cfg = env_cfg,eval_cfg=None)
    env = HistoryWrapper(env) 
    env_pushed = EvalWrapper(env, env_cfg, cmd_vel=cmd_vel,
                      record=False, move_camera=False,experiment_name='Eval')
    
    for force in force_list:
        stationary_push_config = PushConfig(
            id = 0,
            body_index_list = [0, 2,6,10,14, 3,7,11,15, 4,8,12,16],
            change_interval = 100,
            force_list = [force],
        )
        env_pushed.set_eval_config(
            eval_config = [stationary_push_config]
        )
        eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + "-stationary_push-" +baseline_name+ "-" + name + "-force-" + str(force)+ "N"
        for i in range(int(env.max_episode_length) + 10):
            obs_dict = env_pushed.obs_dict
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
                env_pushed.step(actions.detach(), draw=False)

        eval_res = env_pushed.get_result()

        eval_path = os.path.join(eval_path,str(force)+'N')
        if os.path.exists(eval_path) == False:
            os.makedirs(eval_path)
        eval_file_name = os.path.join(eval_path,eval_name)
        np.save(eval_file_name, eval_res)
        print("Done force: ",force)
        env_pushed.reset()
    print("Done")
    

def eval_2Body_multiforce(args, path , cmd_vel =[0.5,0.0,0.0],name='Eval', baseline_name = 'Expert',
                    eval_path = "logs/Eval",
                    force_list = [50,100,150,200,250,300,350,400,450,500]):
    env_cfg = EnvCfg()
    if baseline_name == 'expert':
        train_cfg = ExpertRunnerCfg()
        AC = ExpertActorCritic
    elif baseline_name == 'rma':
        train_cfg = RMARunnerCfg()
        AC = RMAActorCritic
    elif baseline_name == 'e_net':
        train_cfg = EstimatorNetRunnerCfg()
        AC = EstimatorNetActorCritic
    else:
        raise NotImplementedError
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

    policy_cfg  = class_to_dict(train_cfg.policy)
    policy = AC(env.num_obs,
                env.num_privileged_obs,
                env.num_obs_history,
                num_actions=env.num_actions,
                **policy_cfg).to(device)
    policy.load_state_dict(torch.load(path)['model_state_dict'],strict=False)
    policy.eval()
    
    
    env_pushed = EvalWrapper(env, env_cfg, cmd_vel=cmd_vel,
                      record=False, move_camera=False,experiment_name='Eval')
    
    for force in force_list:
        push_config0 = PushConfig(
            id = 0,
            body_index_list = [2,6,10,14, 3,7],
            change_interval = 100,
            force_list = [force],
        )
        push_config1 = PushConfig(
            id = 1,
            body_index_list = [11,15, 4,8,12,16],
            change_interval = 100,
            force_list = [force],
        )
        env_pushed.set_eval_config(
            eval_config = [push_config0,push_config1]
        )
        eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + "-2BodyPush-" +baseline_name+ "-" + name + "-force-" + str(force)+ "N"
        for i in range(int(env.max_episode_length) + 10):
            obs_dict = env_pushed.obs_dict
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
                env_pushed.step(actions.detach(), draw=False)

        eval_res = env_pushed.get_result()

        tmp_eval_path = os.path.join(eval_path,str(force)+'N')
        if os.path.exists(tmp_eval_path) == False:
            os.makedirs(tmp_eval_path)
        eval_file_name = os.path.join(tmp_eval_path,eval_name)
        np.save(eval_file_name, eval_res)
        print("Done force: ",force)
        env_pushed.reset()
    print("Done")
    
def eval_3Body_multiforce(args, path , cmd_vel =[0.5,0.0,0.0],name='Eval', baseline_name = 'Expert',
                    eval_path = "logs/Eval",
                    force_list = [5,10,15,20,25,30,35,40,45,50]):
    env_cfg = EnvCfg()
    if baseline_name == 'expert':
        train_cfg = ExpertRunnerCfg()
        AC = ExpertActorCritic
    elif baseline_name == 'rma':
        train_cfg = RMARunnerCfg()
        AC = RMAActorCritic
    elif baseline_name == 'e_net':
        train_cfg = EstimatorNetRunnerCfg()
        AC = EstimatorNetActorCritic
    else:
        raise NotImplementedError
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

    policy_cfg  = class_to_dict(train_cfg.policy)
    policy = AC(env.num_obs,
                env.num_privileged_obs,
                env.num_obs_history,
                num_actions=env.num_actions,
                **policy_cfg).to(device)
    policy.load_state_dict(torch.load(path)['model_state_dict'],strict=False)
    policy.eval()
    
    
    env_pushed = EvalWrapper(env, env_cfg, cmd_vel=cmd_vel,
                      record=False, move_camera=False,experiment_name='Eval')
    
    for force in force_list:
        push_config0 = PushConfig(
            id = 0,
            body_index_list = [2,6,10,14,],
            change_interval = 100,
            force_list = [force],
        )
        push_config1 = PushConfig(
            id = 1,
            body_index_list = [ 3,7, 11,15],
            change_interval = 100,
            force_list = [force],
        )
        push_config2 = PushConfig(
            id = 2,
            body_index_list = [4,8,12,16],
            change_interval = 100,
            force_list = [force],
        )
        env_pushed.set_eval_config(
            eval_config = [push_config0,push_config1,push_config2]
        )
        eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + "-3BodyPush-" +baseline_name+ "-" + name + "-force-" + str(force)+ "N"
        for i in range(int(env.max_episode_length) + 10):
            obs_dict = env_pushed.obs_dict
            with torch.no_grad():
                actions = policy.act_inference(obs_dict)
                env_pushed.step(actions.detach(), draw=False)

        eval_res = env_pushed.get_result()

        tmp_eval_path = os.path.join(eval_path,str(force)+'N')
        if os.path.exists(tmp_eval_path) == False:
            os.makedirs(tmp_eval_path)
        eval_file_name = os.path.join(tmp_eval_path,eval_name)
        np.save(eval_file_name, eval_res)
        print("Done force: ",force)
        env_pushed.reset()
    print("Done")

if __name__ == '__main__':
    args = get_args()
    path = "logs/EstimatorNet/Dec06_14-19-11_PushBaseline/model_10000.pt"
    # eval_2Body_multiforce(
    #     args = args,
    #     path = path,
    #     cmd_vel =[1.0,0.0,0.0],#!
    #     name='2BodyV10', #!
    #     baseline_name = 'e_net',
    #     eval_path = "logs/Eval/ood/2body/v10", #! 
    #     force_list = [10,20,30,40,50,60,70,80,90,100]
    # )

    eval_3Body_multiforce(
        args = args,
        path = path,
        cmd_vel =[1.0,0.0,0.0],#!
        name='3BodyV10', #!
        baseline_name = 'e_net',
        eval_path = "logs/Eval/ood/3body/v10", #! 
        force_list = [5,10,15,20,25,30,35,40,45,50]
    )
    print("Done")
    exit()