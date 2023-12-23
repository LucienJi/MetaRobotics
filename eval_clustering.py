import os
from datetime import datetime
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from visualization.play_helper import play_policy
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.envs.wrapper.history_wrapper import HistoryWrapper
from isaacgym import gymutil

from OnlineAdaptation.configs.vq_training_config import RunnerCfg
from OnlineAdaptation.configs.eval_config import EnvCfg
from OnlineAdaptation.modules.ac import VQActorCritic
import torch 
from legged_gym.envs.wrapper.push_eval_wrapper import PushConfig,EvalWrapper
import numpy as np 
import argparse

def get_eval_args():
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
        #! For baseline
        {"name": "--baseline_name",'type':str, "default": "expert"},
        {"name": "--task_type",'type':str, "default": "iid"},
        {"name": "--model_path",'type':str, "default": "expert"},
        {"name": "--eval_name",'type':str, "default": "IID"},
        {"name": "--cmd_vel",'type':float, "default": 0.5},
        {"name": "--eval_path", 'type':str, "default": "logs/Eval/ood/3body/v10"},
    ]
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('--force_list', nargs='+', type=int, default=[5,10,15,20,25,30,35,40,45,50])
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

stationary_push_config = PushConfig(
    id = 0,
    body_index_list = [4],
    change_interval = 100,
    force_list = [100],

)


def eval_clustering(args, path , cmd_vel =[0.5,0.0,0.0],name='Eval', model_name = 'Expert',force = 100,
                    eval_path = "logs/Eval"):
    env_cfg = EnvCfg()
    train_cfg = RunnerCfg()
    #! modify args 
    if args.n_heads != -1:
        train_cfg.policy.n_heads = args.n_heads
    if args.use_forward != -1:
        train_cfg.algorithm.use_forward = True if args.use_forward == 1 else False
    if args.stop_gradient != -1:
        train_cfg.algorithm.stop_gradient = True if args.stop_gradient == 1 else False
    if args.codebook_size != -1:
        train_cfg.policy.codebook_size = args.codebook_size

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1000)

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
    policy = VQActorCritic(env.num_obs,env.num_privileged_obs,env.num_obs_history,
                              env.num_history,
                              num_actions=env.num_actions,
                              use_forward=train_cfg.algorithm.use_forward,
                                        **policy_cfg).to(device)
    policy.load_state_dict(torch.load(path)['model_state_dict'],strict=False)
    policy.eval()
    env_pushed = EvalWrapper(env, env_cfg, cmd_vel=cmd_vel,async_mode=True,
                      record=False, move_camera=False,experiment_name='Eval')
    single_body_push_config = PushConfig(
        id = 0,
        # body_index_list = [2,6,10,14, 3,7,11,15, 4,8,12,16],
        body_index_list = [2,6,10,14, 3,7,11,15, ],
        change_interval = -1,
        force_list = [force],
    )
    single_body_push_config._change(num_env=env_cfg.env.num_envs)
    env_pushed.set_eval_config(
            eval_config = [single_body_push_config]
        )
    eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + f"-Clustering_SingleBody-" +model_name+ "-" + name + "-force-" + str(force)+ "N"

    list_indices = []
    list_latents = []
    for i in range(int(env.max_episode_length) + 10):
        obs_dict = env_pushed.obs_dict
        with torch.no_grad():
            actions = policy.act_inference(obs_dict)
            latent, indices, _ = policy.get_VQ_info(obs_dict)
            env_pushed.step(actions.detach(), draw=False)

            list_indices.append(indices.detach().cpu().numpy())
            list_latents.append(latent.detach().cpu().numpy())

    eval_res = env_pushed.get_result()
    list_indices = np.stack(list_indices,axis=1)
    list_latents = np.stack(list_latents,axis=1)

    eval_res['indices'] = list_indices
    eval_res['latents'] = list_latents # (num_envs, num_steps, num_latents)
    
    # tmp_eval_path = os.path.join(eval_path,str(force)+'N')
    tmp_eval_path = eval_path
    if os.path.exists(tmp_eval_path) == False:
        os.makedirs(tmp_eval_path)
    eval_file_name = os.path.join(tmp_eval_path,eval_name)
    np.save(eval_file_name, eval_res)
    print("Done force: ",force)
    env_pushed.reset()
    
    #! 2 body push
    two_body_push_0 = PushConfig(
            id = 1,
            # body_index_list = [2,3,4, 6,7,8 , 10,11,12, 14,15,16],
            body_index_list = [2,3, 6,7 , 10,11, 14,15],
            change_interval = -1,
            force_list = [force],
        )
    two_body_push_1 = PushConfig(
            id = 2,
            body_index_list = [2,3, 6,7 , 10,11, 14,15],
            change_interval = -1,
            force_list = [force],
        )
    env_pushed.set_eval_config([two_body_push_0,two_body_push_1])
    env_pushed.reset() 
    
    eval_name = train_cfg.runner.experiment_name + "-" + train_cfg.runner.run_name + f"-Clustering_TwoBody-" +model_name+ "-" + name + "-force-" + str(force)+ "N"
    list_indices = []
    list_latents = []
    for i in range(int(env.max_episode_length) + 10):
        obs_dict = env_pushed.obs_dict
        with torch.no_grad():
            actions = policy.act_inference(obs_dict)
            latent, indices, _ = policy.get_VQ_info(obs_dict)
            env_pushed.step(actions.detach(), draw=False)

            list_indices.append(indices.detach().cpu().numpy())
            list_latents.append(latent.detach().cpu().numpy())

    eval_res = env_pushed.get_result()

    # tmp_eval_path = os.path.join(eval_path,str(force)+'N')
    tmp_eval_path = eval_path
    if os.path.exists(tmp_eval_path) == False:
        os.makedirs(tmp_eval_path)
    eval_file_name = os.path.join(tmp_eval_path,eval_name)
    np.save(eval_file_name, eval_res)
    print("Done force: ",force)


if __name__ == '__main__':
    args = get_eval_args()
    path = "logs/VQ_Eval/Dec15_14-49-20_VQ/model_10000.pt"

    eval_clustering(
        args,
        path = path,
        cmd_vel=[1.0,0.0,0.0],
        name='Clustering_v10',
        model_name = 'VQ_32',
        force = 20,
        eval_path = "logs/Eval3/clustering"
    )