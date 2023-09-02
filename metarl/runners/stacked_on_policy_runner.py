import time
from collections import deque
import copy
import os
from torch.utils.tensorboard import SummaryWriter
import statistics
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from metarl.modules.StackedAC import ActorCritic,ControlActorCritic,MOActorCritic
from metarl.algorithms.stacked_ppo import PPO
from legged_gym.envs.Go1.quad_config import QuadCfg,QuadRunnerCfg
import json 
from legged_gym.utils import class_to_dict
from .utils import NumpyEncoder
from metarl.algorithms.utils import torch_rand_float 
import numpy as np 
import torch
import pickle

def dump_info( src_dict, tmp_dict):
    for k,v in tmp_dict.items():
        if k not in src_dict.keys():
            src_dict[k] = []
        src_dict[k].append(v)
    return src_dict


class Runner:

    def __init__(self, env:LeggedRobot,log_dir,cfg:QuadRunnerCfg, device='cpu'):
        self.device = device
        self.env = env
        self.train_cfg = cfg 
        self.policy_cfg = class_to_dict(cfg.policy)
        self.cfg = cfg.runner
        self.ppo_cfg = cfg.algorithm

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      **self.policy_cfg,
                                      ).to(self.device)

        self.alg = PPO(actor_critic,cfg = self.ppo_cfg, device=self.device)
        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval
        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.writer = None
        self._need_debug = False
        self.current_learning_iteration = 0
        self.log_dir = log_dir 

        self.env.reset()
    

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_expert=False):
        self.save_cfg()
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)
                    if self.log_dir is not None:
                        # Book keeping
                        if 'train/episode' in infos:
                            ep_infos.append(infos['train/episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        #! 清空历史: 
                        self.env.reset_idx(new_ids[:, 0])
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0


                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

            mean_value_loss, mean_surrogate_loss,mean_entropy_loss, mean_adaptation_module_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                if hasattr(self.env, "curricula"):
                    path = os.path.join(self.log_dir,'curricula.pkl')
                    with open( path, "wb" ) as f:
                        pickle.dump( self.env.curricula, f)
                    print("Save Curricula to: ",path) 


            if self.log_dir is not None:
                self.log(locals())
            ep_infos.clear()
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def save_cfg(self):
        # 保存训练使用的cfg参数
        env_cfg_dict = class_to_dict(self.env.cfg) if type(self.env.cfg) is not dict else self.env.cfg
        train_cfg_dict = class_to_dict(self.train_cfg) if type(self.train_cfg) is not dict else self.train_cfg
        env_cfg_json = json.dumps(env_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        train_cfg_json = json.dumps(train_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        os.makedirs(self.log_dir)
        with open(os.path.join(self.log_dir, "env_cfg.json"), 'w') as f:
            f.write(env_cfg_json)
        with open(os.path.join(self.log_dir, "train_cfg.json"), 'w') as f:
            f.write(train_cfg_json)
            
    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']
    

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/Adaptation_module', locs['mean_adaptation_module_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        if self._need_debug:
            macro_score_string = ""
            for k,v in locs['macro_scores'].items():
                self.writer.add_scalar(f"Macro/{k}", v, locs['it'])
                macro_score_string += f"""{f'Mean macro {k}:':>{pad}} {v:.4f}\n"""
        else:
            macro_score_string = None
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Adaptation loss:':>{pad}} {locs['mean_adaptation_module_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Policy entropy:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Adaptation loss:':>{pad}} {locs['mean_adaptation_module_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Policy entropy:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += macro_score_string if macro_score_string is not None else ""
        
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
    
    def hot_update(self):
        with open("hot_params/expert_hot.json") as f:
            hot_params = json.load(f)
        if hot_params['hot_update']:
            #! 仅支持 entropy ceof 
            new_entropy_coef = hot_params['entropy_coef']
            self.alg.entropy_coef = new_entropy_coef
            print("Hot Update entropy: ",new_entropy_coef)

class EvalRunner:
    def __init__(self,
                 env:LeggedRobot,
                 use_style:bool,
                 expert_model_path,
                 cfg,
                 log_dir=None,
                 device='cpu'):
        
        self.device = device
        self.env = env
        self.num_envs = self.env.num_envs
        self.env.eval = True
        self.train_cfg = cfg 
        self.policy_cfg = class_to_dict(cfg.policy)
        self.cfg = cfg.runner
        self.ppo_cfg = cfg.algorithm
        self.sampler_cfg = class_to_dict(cfg.sampler)
        self.sampler_baseline = cfg.sampler.baseline
        self.use_control = self.ppo_cfg.use_control  if hasattr(self.ppo_cfg,'use_control') else True
        self.use_style = use_style 
        self.reward_dim = self.env.num_unfixed_rewards
        if self.use_style:
            if self.use_control:
                expert = ActorCritic(self.env.num_obs,
                                        self.env.num_privileged_obs,
                                        self.env.num_obs_history,
                                        self.env.num_actions,
                                        **self.policy_cfg)
                if expert_model_path is not None: 
                    self.load_expert(expert,expert_model_path)
                    print("Load Exeprt From: ", expert_model_path)
                actor_critic = ControlActorCritic(expert,self.env.num_obs,
                                        self.env.num_privileged_obs,
                                        self.env.num_obs_history,
                                        self.reward_dim,
                                        self.env.num_actions,
                                        **self.policy_cfg
                                        ).to(self.device)
            else:
                actor_critic = MOActorCritic(self.env.num_obs,
                                        self.env.num_privileged_obs,
                                        self.env.num_obs_history,
                                        self.reward_dim,
                                        self.env.num_actions,
                                        **self.policy_cfg
                                        ).to(self.device)
        else:
            actor_critic = ActorCritic(self.env.num_obs,
                                        self.env.num_privileged_obs,
                                        self.env.num_obs_history,
                                        self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
     
        if use_style:
            self.alg = MO_PPO(actor_critic, device=self.device, cfg=self.ppo_cfg)
        else:
            self.alg = PPO(actor_critic,device=self.device,cfg=self.ppo_cfg)

        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval

        # init sampler and model 
        if use_style:
            preference_net = PreferenceNet(
            preference_dim=self.env.num_styles,
            reward_dim= self.env.num_unfixed_rewards,
            hidden_dim=512
            )
            self.sampler = Sampler_algo(
                preferencenet=preference_net,
                device = self.device, 
                **self.sampler_cfg
            )
            self.sampler.init_dataset(reward_dim = self.env.num_unfixed_rewards,
                                    style_dim=self.env.num_styles,
                                    preference_net=preference_net,min_score=self.env.cfg.env.min_score,max_score=self.env.cfg.env.max_score)
        else:
            self.preference_net, self.sampler = None, None 
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.env.set_eval()
        self.env.reset()
        self.alg.test_mode()
    
    def view(self, num_iterations, num_steps , cmd_vel = None, cmd_preference = None, cmd_scale = None):
        self.env.eval = True 
        self.alg.test_mode()
        self.alg.actor_critic.eval()
        if cmd_vel is not None:
            commanded_vel = torch.tensor(cmd_vel).to(self.env.device).squeeze()
            self.env.commands[:,0:3] = commanded_vel 
        if self.use_style:
            if cmd_preference is None:
                if cmd_scale is None:
                    scales = self.sampler.sample_scales(self.env.num_envs, random= True)
                else:
                    scales = torch.tensor(cmd_scale).to(self.env.device).squeeze() 
                    scales = scales.unsqueeze(0).repeat((self.env.num_envs,1))
                self.env.update_reward_weights(scales.to(self.env.device), discrete=True )#! 其实这步有点多余了
            else:
                cmd_preference = torch.tensor(cmd_preference).to(self.env.device).squeeze() 
                cmd_preference = cmd_preference.unsqueeze(0).repeat((self.env.num_envs,1))
                scales = self.sampler.act(cmd_preference)
                print("Scale Check: ", scales.mean(0).cpu().numpy())
                self.env.update_reward_weights(scales.to(self.env.device), discrete=True )#! 其实这步有点多余了
        with torch.inference_mode():
            self.env.reset()

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)

        for it in range(1, num_iterations + 1):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(num_steps):
                    if self.use_style:
                        actions = self.alg.actor_critic.act_student(obs_history,scales)
                    else:
                        actions = self.alg.actor_critic.act_student(obs_history)
                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]
                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                stop = time.time()
                collection_time = stop - start
                # Learning step
                start = stop
        print("Done")
    
    def eval_sequential(self,num_iter, max_step, commanded_vel, change_points:list, 
                        commanded_prefernce_list = None,
                        eval_name = 'eval_seq'):
        self.env.set_eval()
        with torch.inference_mode():
           self.env.reset()
        # set x,y,omega,heading 
        if not isinstance(commanded_vel[0],list):
            commanded_vel = [commanded_vel]

        if 0 in change_points:
            change_points.remove(0)
        change_points = sorted(change_points)
        n_change_points = len(change_points)
        scales_list,scales_list_tosave = [],[]
        if self.use_style:
            if commanded_prefernce_list is None:
                commanded_prefernce_list = self.sampler.sample_scales(n_change_points, random = True) 
            n_cmd_preference = len(commanded_prefernce_list)
            for i in range(n_change_points):
                scale = commanded_prefernce_list[i%n_cmd_preference].unsqueeze(0).repeat((self.env.num_envs,1)) 
                scales_list.append(scale)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        eval_dict = {}
        self.alg.actor_critic.eval() 
        for it in range(num_iter):
            ptr = 0 
            with torch.inference_mode():
                for i in range(max_step):
                    cmd_vel = commanded_vel[ptr % len(commanded_vel)]
                    self.env.set_command(cmd_vel)
                    if self.use_style:
                        scales = scales_list[ptr % len(scales_list)]
                        scales_list_tosave.append(scales.cpu().numpy())
                        actions = self.alg.actor_critic.act_inference(obs_dict,scales)
                    else:
                        actions = self.alg.actor_critic.act_inference(obs_dict)
                    
                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                   
                    if self.log_dir is not None:
                        # Book keeping
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        cur_episode_length[new_ids] = 0
                    dump_info(eval_dict,self.env.get_eval_data())
                    if i > change_points[ptr] and ptr < (len(change_points)-1):
                        ptr += 1
        if self.use_style:
            eval_dict['scales'] = np.stack(scales_list_tosave,axis=1)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        np.savez(os.path.join(self.log_dir,eval_name+'.npz'),**eval_dict)
        print("Eval Done")


    def eval(self,num_iter,max_step,commanded_vel, commanded_prefernce = None , eval_name = 'Eval'):
        """
        1. 定速测试 
        """
        self.env.set_eval()
        with torch.inference_mode():
            self.env.reset()
        ## 设定速度 
        self.env.set_command(commanded_vel)
        preference = None 
        if self.use_style:
            
            if commanded_prefernce is None:
                scales = self.sampler.sample_scales(self.env.num_envs, random= True)
            else:
                commanded_prefernce = torch.tensor(commanded_prefernce).to(self.env.device).squeeze() 
                preference = commanded_prefernce
                scales = self.sampler.act(commanded_prefernce,use_model=False)
                scales = scales.repeat((self.env.num_envs,1))
        
        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        self.alg.actor_critic.eval() 

        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        total_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        total_episode_count = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        eval_dict = {}
        lenbuffer = deque(maxlen=100)

        for it in range(num_iter):
            with torch.inference_mode():
                for i in range(max_step):
                    if self.use_style:
                        actions = self.alg.actor_critic.act_inference(obs_dict,scales)
                    else:
                        actions = self.alg.actor_critic.act_inference(obs_dict)
                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                    if self.log_dir is not None:
                        # Book keeping
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        total_episode_length[new_ids] += cur_episode_length[new_ids]
                        total_episode_count[new_ids] += 1 
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_episode_length[new_ids] = 0
                    
                    dump_info(eval_dict,self.env.get_eval_data())

        episode_length = total_episode_length / (total_episode_count + 1e-6)
        episode_length = episode_length.cpu().numpy()
        eval_dict['episode_length'] = episode_length
        if self.use_style:
            eval_dict['scale'] = scales.cpu().numpy()
        if preference is not None:
            eval_dict['preference'] = preference.cpu().numpy()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        np.savez(os.path.join(self.log_dir,eval_name+'.npz'),**eval_dict)
        print("Eval Done")

    def eval_pf(self,num_iter,max_step,commanded_vel, use_sampled_preference =False, use_target_scales = None, eval_name = "Eval_PF"):
        self.env.set_eval()
        with torch.inference_mode():
            self.env.reset()
        self.env.set_command(commanded_vel)
        macro_scores = dict()
        for it in range(num_iter):
            preference = None 
            if self.use_style:
                if use_sampled_preference:
                    preference, scales = self.sampler.sample_eval_weights(self.num_envs, use_model=False)
                else:
                    preference = None 
                    if use_target_scales is not None:
                        scales =  torch.tensor(use_target_scales).to(self.env.device).unsqueeze(0).repeat(self.num_envs,1)
                    else:
                        scales = self.sampler.sample_scales(self.env.num_envs, random= True)
            with torch.inference_mode():
                self.env.reset() 
            eval_dict = {}
            obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
            with torch.inference_mode():
                for i in range(max_step):
                    if self.use_style:
                        actions = self.alg.actor_critic.act_inference(obs_dict,scales)
                    else:
                        actions = self.alg.actor_critic.act_inference(obs_dict)

                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                    dump_info(eval_dict,self.env.get_eval_data())
            for k, v in eval_dict.items():
                eval_dict[k] = np.stack(v, axis=0)
            tmp_macro_scores = MacroScoreManager.eval_macro_score(eval_dict) 
            if self.use_style:
                tmp_macro_scores['scale'] = scales.cpu().numpy() # n_env, n_style 
                if preference is not None:
                    tmp_macro_scores['preference'] = preference.cpu().numpy()
            dump_info(macro_scores,tmp_macro_scores)

        for k,v in macro_scores.items():
            macro_scores[k] = np.concatenate(v,axis=0)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        np.savez(os.path.join(self.log_dir,eval_name+'.npz'),**macro_scores)
        print("Eval Done")

    
    def save(self, path, infos=None):
        if self.use_style:
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'preference':self.sampler.preferencenet.state_dict(),
                'sampler_optimizer_state_dict':self.sampler.optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos,
                }, path)
        else:
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos,
                }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if self.use_style:
            self.sampler.preferencenet.load_state_dict(loaded_dict['preference'])

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if self.use_style:
                self.sampler.optimizer.load_state_dict(loaded_dict['sampler_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def load_expert(self,expert_model:ActorCritic,path):
        loaded_dict = torch.load(path)
        expert_model.load_state_dict(loaded_dict['model_state_dict'])
    def load_dataset(self,dir):
        if hasattr(self,'sampler'):
            self.sampler.load_database(dir)
            print("Load Database From: ", dir)
    def save_dataset(self,dir):
        if hasattr(self,'sampler'):
            self.sampler.save_database(dir)
            print("Save Database To: ", dir)