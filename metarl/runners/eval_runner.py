
import time
import os
from collections import deque
import statistics
from typing import Union    
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np 
from MORL_v1.algorithms.mo_ppo import MO_PPO
from MORL_v1.algorithms.ppo import PPO
from MORL_v1.modules.MOAC import MOActorCritic
from MORL_v1.modules.ControlAC import ControlActorCritic,ActorCritic
from .utils import VecEnv
from legged_gym.envs.quad.mo_quad import MOQuadRobot
from MORL_v1.algorithms.Sampler import Sampler_algo 
from MORL_v1.modules.Preference import PreferenceNet as PreferenceNet
import json 
from legged_gym.utils import class_to_dict
from .utils import NumpyEncoder
class MO_EvalOnPolicyRunner:

    def __init__(self,
                 env: MOQuadRobot,
                 use_style:bool,
                 expert_model_path,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.train_cfg = train_cfg
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        if use_style:
            self.sampler_cfg = train_cfg["sampler"]

        self.device = device
        self.env = env
        self.num_envs = self.env.num_envs
        self._debug = self.env._debug
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        num_unfixed_rewards = self.env.num_unfixed_rewards
        if use_style and expert_model_path is not None:
            expert = ActorCritic(self.env.num_obs,num_critic_obs,self.env.num_actions,**self.policy_cfg).to(self.device)
            self.load_expert(expert,expert_model_path)
            print("Load Exeprt From: ", expert_model_path)
            actor_critic = ControlActorCritic(expert, self.env.num_obs,num_critic_obs,num_unfixed_rewards,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        else:
            actor_critic = ActorCritic(self.env.num_obs,num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        
        if use_style:
            self.alg = MO_PPO(actor_critic, device=self.device, **self.alg_cfg)
        else:
            self.alg = PPO(actor_critic,device=self.device,**self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init sampler and model 
        if use_style:
            preference_net = PreferenceNet(
                preference_dim=self.env.num_styles,
                reward_dim= num_unfixed_rewards,
                hidden_dim=512
            )
            self.sampler = Sampler_algo(
                preferencenet=preference_net,
                device = self.device, 
                **self.sampler_cfg
            )

            self.sampler.init_dataset(reward_dim = num_unfixed_rewards,
                                    style_dim=self.env.num_styles,
                                    preference_net=preference_net)
        else:
            self.preference_net, self.sampler = None, None 
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.use_style = use_style 

        _, _ = self.env.reset()
    
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
                self.env._update_reward_weights(scales.to(self.env.device), discrete=True )#! 其实这步有点多余了
            else:
                cmd_preference = torch.tensor(cmd_preference).to(self.env.device).squeeze() 
                cmd_preference = cmd_preference.unsqueeze(0).repeat((self.env.num_envs,1))
                scales = self.sampler.act(cmd_preference)
                print("Scale Check: ", scales.mean(0).cpu().numpy())
                self.env._update_reward_weights(scales.to(self.env.device), discrete=True )#! 其实这步有点多余了
        with torch.inference_mode():
            _, _ = self.env.reset()

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        for it in range(1, num_iterations + 1):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(num_steps):
                    if self.use_style:
                        actions = self.alg.inference(obs,critic_obs,scales)
                    else:
                        actions = self.alg.inference(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                stop = time.time()
                collection_time = stop - start
                # Learning step
                start = stop
        print("Done")
    
    def eval_sequential(self,num_iter, max_step, commanded_vel, change_points:list, 
                        commanded_prefernce_list = None,
                        eval_name = 'eval_seq'):
        self.env.eval = True 
        with torch.inference_mode():
            _, _ = self.env.reset()
        # set x,y,omega,heading 
        commanded_vel = torch.tensor(commanded_vel).to(self.env.device).squeeze()
        self.env.commands[:,0:3] = commanded_vel 

        if 0 in change_points:
            change_points.remove(0)
        change_points = sorted(change_points)
        n_change_points = len(change_points)
        scales = []

        if self.use_style:
            if commanded_prefernce_list is None:
                commanded_prefernce_list = self.sampler.sample_scales(n_change_points, random = True) 
            last_point = 0
            for i in range(n_change_points):
                ptr = change_points[i]
                scale = commanded_prefernce_list[i].unsqueeze(0).repeat((ptr - last_point,1)) 
                scales.append(scale)
                last_point = ptr 
            scales_list = torch.concat(scales,dim = 0)
            n_scales = len(scales_list)
            
        
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        steps = []
        lin_vel = []
        ang_vel = []
        cmd_vel = []
        contacts = []


        macro_steps = []
        macro_scores_dict = {}

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.eval() 
        for it in range(num_iter):
            with torch.inference_mode():
                for i in range(max_step):
                    if self.use_style:
                        scales = scales_list[cur_episode_length%n_scales]
                        actions = self.alg.act(obs, critic_obs,scales)
                    else:
                        actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.env.update_macro_score()
                    steps.append(cur_episode_length.cpu().numpy())
                    lin_vel.append(self.env.base_lin_vel[:,0:2].cpu().numpy())
                    ang_vel.append(self.env.base_ang_vel[:, 2].cpu().numpy())
                    cmd_vel.append(self.env.commands[:,0:3].cpu().numpy())
                    contacts.append(self.env.contacts.cpu().numpy())

                    
                    if i > 0 and (i % 50 == 0):
                        macro_steps.append(cur_episode_length.cpu().numpy())
                        macro_scores = self.env.get_macro_score() # num_envs,
                        for k,v in macro_scores.items():
                            if k not in macro_scores_dict.keys():
                                macro_scores_dict[k] = []
                            macro_scores_dict[k].append(v.squeeze().cpu().numpy())

                    if self.log_dir is not None:
                        # Book keeping
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        cur_episode_length[new_ids] = 0
        
        steps = np.stack(steps,axis=0)
        lin_vel = np.stack(lin_vel,axis=0)
        ang_vel = np.stack(ang_vel,axis=0)
        cmd_vel = np.stack(cmd_vel,axis=0)
        macro_steps = np.stack(macro_steps,axis=0)
        contacts = np.stack(contacts,axis=0)
        for k,v in macro_scores_dict.items():
            macro_scores_dict[k] = np.stack(v,axis=0)
        macro_scores_dict['lin_vel'] = lin_vel 
        macro_scores_dict['ang_vel'] = ang_vel 
        macro_scores_dict['cmd_vel'] = cmd_vel 
        macro_scores_dict['steps'] = steps 
        macro_scores_dict['macro_steps'] = macro_steps
        macro_scores_dict['contacts'] = contacts
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        np.savez(os.path.join(self.log_dir,eval_name+'.npz'),**macro_scores_dict)
        print("Eval Done")


    def eval(self,num_iter,max_step,commanded_vel, commanded_prefernce = None , eval_name = 'Eval'):
        self.env.eval = True 
        with torch.inference_mode():
            _, _ = self.env.reset()

        # set x,y,omega,heading 
        commanded_vel = torch.tensor(commanded_vel).to(self.env.device).squeeze()
        self.env.commands[:,0:3] = commanded_vel 
        if self.use_style:
            if commanded_prefernce is None:
                scales = self.sampler.sample_scales(self.env.num_envs, random= True)
                self.env._update_reward_weights(scales.to(self.env.device), discrete=True )#! 其实这步有点多余了
            else:
                commanded_prefernce = torch.tensor(commanded_prefernce).to(self.env.device).squeeze() 
                commanded_prefernce = commanded_prefernce.unsqueeze(0).repeat((self.env.num_envs,1))
                scales = self.sampler.act(commanded_prefernce)
                self.env._update_reward_weights(scales.to(self.env.device), discrete=True )#! 其实这步有点多余了
        
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.eval() 

        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        total_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        total_episode_count = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        total_lin_vel = torch.zeros((self.env.num_envs,2), dtype=torch.float, device=self.device)
        total_ang_vel = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        total_vel_count = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        macro_score_dict = {}

        lenbuffer = deque(maxlen=100)

        for it in range(num_iter):
            with torch.inference_mode():
                for i in range(max_step):
                    if self.use_style:
                        actions = self.alg.act(obs, critic_obs,scales)  
                    else:
                        actions = self.alg.act(obs,critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.env.update_macro_score()
                    total_lin_vel[:,0:2] += self.env.base_lin_vel[:,0:2].to(self.device)
                    total_ang_vel[:] += self.env.base_ang_vel[:, 2].to(self.device)
                    total_vel_count[:] += 1.0 

                    if self.log_dir is not None:
                        # Book keeping
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        total_episode_length[new_ids] += cur_episode_length[new_ids]
                        total_episode_count[new_ids] += 1 
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_episode_length[new_ids] = 0
                macro_scores = self.env.get_macro_score() # num_envs,
                for k,v in macro_scores.items():
                    if k not in macro_score_dict.keys():
                        macro_score_dict[k] = []
                    macro_score_dict[k].append(v.squeeze().cpu().numpy())
        
        episode_length = total_episode_length / total_episode_count
        episode_length = episode_length.cpu().numpy()
        lin_vel = (total_lin_vel / total_vel_count.unsqueeze(-1)).cpu().numpy()
        ang_vel = (total_ang_vel / total_vel_count).cpu().numpy() 

        for k,v in macro_score_dict.items():
            avg_score = np.stack(v,axis = -1)
            avg_score = np.mean(avg_score,axis=-1)
            macro_score_dict[k] = avg_score 
        macro_score_dict['episode_length'] = episode_length 
        macro_score_dict['lin_vel']  = lin_vel 
        macro_score_dict['ang_vel'] = ang_vel 
        macro_score_dict['commanded_vel'] = self.env.commands[:,0:3].cpu().numpy()
        if self.use_style:
            macro_score_dict['scale'] = scales.cpu().numpy()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        np.savez(os.path.join(self.log_dir,eval_name+'.npz'),**macro_score_dict)
        print("Eval Done")

    def eval_pf(self,num_iter,max_step,commanded_vel, use_sampled_preference =False, use_target_scales = None, eval_name = "Eval_PF"):
        self.env.eval = True 
        self.alg.test_mode()
        commanded_vel = torch.tensor(commanded_vel).to(self.env.device).squeeze()
        self.env.commands[:,0:3] = commanded_vel 
        self.alg.test_mode()
        macro_score_dict = {}
        macro_score_dict['episode_length'] = []
        for it in range(num_iter):
            if use_sampled_preference:
                preference,scales = self.sampler.sample_eval_weights(self.num_envs)
            else:
                if use_target_scales is not None:
                    scales =  torch.tensor(use_target_scales).to(self.env.device).unsqueeze(0).repeat(self.num_envs,1)
                else:
                    scales = self.sampler.sample_scales(self.env.num_envs, random= True)
            self.env._update_reward_weights(scales,discrete=True)

            with torch.inference_mode():
                _, _ = self.env.reset() 
                self.env.reset_macro_score()

            obs = self.env.get_observations()
            privileged_obs = self.env.get_privileged_observations()
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

            cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            total_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            total_episode_count = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)


            with torch.inference_mode():
                for i in range(max_step):
                    if self.use_style:
                        actions = self.alg.act(obs, critic_obs,scales)  
                    else:
                        actions = self.alg.act(obs,critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.env.update_macro_score()
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    total_episode_length[new_ids] += cur_episode_length[new_ids]
                    total_episode_count[new_ids] += 1 
                    cur_episode_length[new_ids] = 0

                macro_scores = self.env.get_macro_score() # num_envs,
                for k,v in macro_scores.items():
                    if k not in macro_score_dict.keys():
                        macro_score_dict[k] = []
                    if k != 'scale':
                        macro_score_dict[k].append(v.squeeze().cpu().numpy())
                    else:
                        macro_score_dict[k].append(scales.squeeze().cpu().numpy())
                episode_length = total_episode_length / (total_episode_count + 1e-6)
                episode_length = episode_length.cpu().numpy()
                macro_score_dict['episode_length'].append(episode_length)

        for k,v in macro_score_dict.items():
            macro_score_dict[k] = np.concatenate(v,axis = 0 ) 
        macro_score_dict['cmd_vel'] = commanded_vel.cpu().numpy()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        np.savez(os.path.join(self.log_dir,eval_name+'.npz'),**macro_score_dict)
        print("Eval Done")

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
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time'] + locs['sampler_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/preference', locs['preference_error'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        self.writer.add_scalar('Perf/sampler_time', locs['sampler_time'], locs['it'])
        
        if 'global_H' in locs['eval_info']:
            gloabl_maxScore = locs['eval_info']['gloabl_maxScore']
            global_minScore = locs['eval_info']['global_minScore']
            self.writer.add_scalar("Sampler/N_samples_in_PF", locs['eval_info']['N_samples_in_PF'], locs['it'])
            self.writer.add_scalar('Sampler/global_H', locs['eval_info']['global_H'], locs['it'])
            self.writer.add_scalar('Sampler/global_SP', locs['eval_info']['global_SP'], locs['it'])
            self.writer.add_scalar('Sampler/local_maxH', locs['eval_info']['local_maxH'], locs['it'])
            self.writer.add_scalar('Sampler/local_minH', locs['eval_info']['local_minH'], locs['it'])
            self.writer.add_scalar('Sampler/local_meanH', locs['eval_info']['local_meanH'], locs['it'])
            self.writer.add_scalar('Sampler/local_maxSP', locs['eval_info']['local_maxSP'], locs['it'])
            self.writer.add_scalar('Sampler/local_minSP', locs['eval_info']['local_minSP'], locs['it'])
            self.writer.add_scalar('Sampler/local_meanSP', locs['eval_info']['local_meanSP'], locs['it'])
            self.writer.add_scalar('Sampler/local_maxN', locs['eval_info']['local_maxN'], locs['it'])
            self.writer.add_scalar('Sampler/local_minN', locs['eval_info']['local_minN'], locs['it'])
            self.writer.add_scalar('Sampler/local_meanN', locs['eval_info']['local_meanN'], locs['it'])
            for i in range(len(gloabl_maxScore)):
                self.writer.add_scalar(f'Sampler/global_maxScore{i}', gloabl_maxScore[i], locs['it'])
                self.writer.add_scalar(f'Sampler/global_minScore{i}', global_minScore[i], locs['it'])
            gloabl_maxScore = list(map(lambda x :str(x) + '%',gloabl_maxScore.round(2)))
            global_minScore = list(map(lambda x :str(x) + '%',global_minScore.round(2)))
            sampler_string = (  f"""{'N_samples_in_PF:':>{pad}} {locs['eval_info']['N_samples_in_PF']:.4f}\n"""
                                f"""{'global_H:':>{pad}} {locs['eval_info']['global_H'].item():.2f}\n"""
                                f"""{'global_SP:':>{pad}} {locs['eval_info']['global_SP'].item():.4f}\n"""
                                f"""{'gloabl_maxScore:':>{pad}} {gloabl_maxScore}\n"""
                                f"""{'global_minScore:':>{pad}} {global_minScore}\n"""
                                f"""{'local_maxH:':>{pad}} {locs['eval_info']['local_maxH'].item():.4f}\n"""
                                f"""{'local_minH:':>{pad}} {locs['eval_info']['local_minH'].item():.4f}\n"""
                                f"""{'local_meanH:':>{pad}} {locs['eval_info']['local_meanH'].item():.4f}\n"""
                                f"""{'local_maxSP:':>{pad}} {locs['eval_info']['local_maxSP'].item():.4f}\n"""
                                f"""{'local_minSP:':>{pad}} {locs['eval_info']['local_minSP'].item():.4f}\n"""
                                f"""{'local_meanSP:':>{pad}} {locs['eval_info']['local_meanSP'].item():.4f}\n"""
                                f"""{'local_maxN:':>{pad}} {locs['eval_info']['local_maxN'].item():.4f}\n"""
                                f"""{'local_minN:':>{pad}} {locs['eval_info']['local_minN'].item():.4f}\n"""
                                f"""{'local_meanN:':>{pad}} {locs['eval_info']['local_meanN'].item():.4f}\n""")
        else:
            sampler_string = None    
        
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        log_string = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{log_string.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s, Sampling {locs['sampler_time']})\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Policy entropy:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'Preference loss:':>{pad}} {locs['preference_error']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{log_string.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s, Sampling {locs['sampler_time']})\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Policy entropy:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                          f"""{'Preference loss:':>{pad}} {locs['preference_error']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        if sampler_string is not None:
            log_string += sampler_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'preference':self.sampler.preferencenet.state_dict(),
            'sampler_optimizer_state_dict':self.sampler.optimizer.state_dict(),
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

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def load_expert(self,expert_model:ActorCritic,path):
        loaded_dict = torch.load(path)
        expert_model.load_state_dict(loaded_dict['model_state_dict'])
    
    def hot_update(self):
        if self.sampler_cfg['baseline']:
            with open("hot_params/baseline_hot.json") as f:
                hot_params = json.load(f)
        else:
            with open("hot_params/morl_hot.json") as f:
                hot_params = json.load(f)
        if hot_params['hot_update']:
            #! 仅支持 entropy ceof 
            new_entropy_coef = hot_params['entropy_coef']
            self.alg.entropy_coef = new_entropy_coef
            print("Hot Update entropy: ",new_entropy_coef)
    
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

        