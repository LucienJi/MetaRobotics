# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os,json 
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from utils.torch_utils import VecEnv,class_to_dict,dump_info,NumpyEncoder
from AMP.configs.training_config import RunnerCfg

from AMP.algorithms.ppo import PPO
from AMP.modules.ac import ActorCritic
from AMP.modules.discriminator import AMPDiscriminator

from AMP.dataset.motion_loader import AMPLoader
from utils.torch_utils import Normalizer

class Runner:

    def __init__(self,
                 env:VecEnv,log_dir,cfg:RunnerCfg, device='cpu'):

        self.device = device
        self.env = env
        self.train_cfg = cfg 
        self.policy_cfg = cfg.policy
        self.cfg = cfg.runner
        self.ppo_cfg = cfg.algorithm
        self.amp_cfg = cfg.amp 
        
        policy_cfg = class_to_dict(self.policy_cfg)
        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      **policy_cfg,
                                      ).to(self.device)

        amp_data = AMPLoader(
            device, 
            time_between_frames=self.env.dt, 
            preload_transitions=True,
            num_preload_transitions=self.train_cfg.amp.amp_num_preload_transitions,
            motion_files=self.train_cfg.amp.amp_motion_files)
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.train_cfg.amp.amp_reward_coef,
            self.train_cfg.amp.amp_discr_hidden_dims,self.device,
            self.train_cfg.amp.amp_task_reward_lerp).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        self.alg = PPO(actor_critic,
                       discriminator=discriminator,
                       amp_data=amp_data,
                       amp_normalizer=amp_normalizer,cfg = self.ppo_cfg, device=self.device)
        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.policy_cfg.num_history,self.env.num_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()
    
    def learn(self, num_learning_iterations):
        self.save_cfg()
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        obs_dict = self.env.get_observations()
        for k,v in obs_dict.items():
            obs_dict[k] = v.to(self.device) 
        amp_obs = self.env.get_amp_observations()
        amp_obs = amp_obs.to(self.device)
        
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

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
                    actions = self.alg.act(obs_dict["obs"],obs_dict["privileged_obs"], obs_dict["obs_history"],
                                           amp_obs= amp_obs)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()
                    next_amp_obs = next_amp_obs.to(self.device)
                    for k,v in obs_dict.items():
                        obs_dict[k] = v.to(self.device)
                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids,terminal_amp_states = infos['reset_env_ids'],infos['terminal_amp_states']
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    
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
                self.alg.compute_returns(obs_dict['obs'], obs_dict['privileged_obs'])

            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

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
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
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

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
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
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save_cfg(self):
        # 保存训练使用的cfg参数
        env_cfg_dict = class_to_dict(self.env.cfg) if type(self.env.cfg) is not dict else self.env.cfg
        train_cfg_dict = class_to_dict(self.train_cfg) if type(self.train_cfg) is not dict else self.train_cfg
        env_cfg_json = json.dumps(env_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        train_cfg_json = json.dumps(train_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(os.path.join(self.log_dir, "env_cfg.json"), 'w') as f:
            f.write(env_cfg_json)
        with open(os.path.join(self.log_dir, "train_cfg.json"), 'w') as f:
            f.write(train_cfg_json)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference