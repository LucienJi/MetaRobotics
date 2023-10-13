from Lips.modules.ac import LipsActorCritic as ActorCritic
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Lips.configs.training_config import RunnerCfg
from Lips.buffers.storage import RolloutStorage

class PPO:
    def __init__(self, actor_critic:ActorCritic,cfg:RunnerCfg.algorithm, device='cpu'):

        self.device = device
        self.cfg = cfg
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        #! 这个是可以优化 expert 的优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = self.cfg.learning_rate
        self.entropy_coef = self.cfg.entropy_coef 

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act_student(self, obs, privileged_obs, obs_history,stage = 1):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, privileged_obs, obs_history,stage).detach()
        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.cfg.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self,stage = 1):
        mean_value_loss = 0
        mean_entropy_loss = 0
        mean_surrogate_loss = 0
        mean_k_sl_loss = 0
        mean_penality = 0 
        mean_l2_loss = 0 

        mean_max_jacob = 0
        mean_min_jacob = 0
        mean_mean_jacob = 0
        
        generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch,\
            target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, env_bins_batch in generator:

            #! Training  Expert 
            self.actor_critic.act(obs_batch, privileged_obs_batch, obs_history_batch,stage)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch, privileged_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.cfg.desired_kl != None and self.cfg.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.cfg.clip_param,
                                                                               1.0 + self.cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-self.cfg.clip_param,
                                                                          self.cfg.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            if stage == 2:
                #! Supervised Learning for K 
                sl_loss = self.actor_critic.actor.sl_k_loss(obs_batch) 
                loss += sl_loss['loss']
                mean_k_sl_loss += sl_loss['sl_loss']
                mean_penality += sl_loss['penality']
                mean_max_jacob += sl_loss['max_jacob']
                mean_min_jacob += sl_loss['min_jacob']
                mean_mean_jacob += sl_loss['mean_jacob']
            elif stage == 1:
                jacob_info = self.actor_critic.actor.raw_jacob_check(obs_batch,ord = 2)
                mean_max_jacob += jacob_info['max_jacob']
                mean_min_jacob += jacob_info['min_jacob']
                mean_mean_jacob += jacob_info['mean_jacob']

            elif stage == 3:
                #! Lips Loss 
                reg_loss = self.actor_critic.actor.reg_loss(obs_batch)
                loss += reg_loss['loss']
                mean_l2_loss += reg_loss['l2_loss']
                mean_penality += reg_loss['penality']
                mean_max_jacob += reg_loss['max_jacob']
                mean_min_jacob += reg_loss['min_jacob']
                mean_mean_jacob += reg_loss['mean_jacob']

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()


    
        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_k_sl_loss /= num_updates
        mean_l2_loss /= num_updates
        mean_penality /= num_updates
        mean_max_jacob /= num_updates
        mean_min_jacob /= num_updates
        mean_mean_jacob /= num_updates

      
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss,mean_entropy_loss,\
            mean_k_sl_loss,mean_l2_loss,mean_penality,\
            mean_max_jacob,mean_min_jacob,mean_mean_jacob
