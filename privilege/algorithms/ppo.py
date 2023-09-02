import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from privilege.buffers.rollout_storage import RolloutStorage
from privilege.modules.ac import ActorCritic    
from legged_gym.envs.configs.basic_config import BasicCfg,BasicRunnerCfg

class PPO:
    def __init__(self, actor_critic:ActorCritic,cfg:BasicRunnerCfg.algorithm, device='cpu'):

        self.device = device
        self.cfg = cfg
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later

        #! 
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate)

        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.adaptation_module.parameters(),
                                                      lr=self.cfg.adaptation_module_learning_rate)

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

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_history,obs).detach()
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
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.cfg.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self):
        mean_value_loss = 0
        mean_entropy_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        
        generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch in generator:

            self.actor_critic.act(obs_history_batch,obs_batch)
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

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step

            # for epoch in range(self.cfg.num_adaptation_module_substeps):

            #     adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
            #     with torch.no_grad():
            #         adaptation_target = privileged_obs_batch

            #     adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)
            #     self.adaptation_module_optimizer.zero_grad()
            #     adaptation_loss.backward()
            #     self.adaptation_module_optimizer.step()

            #     mean_adaptation_module_loss += adaptation_loss.item()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        # mean_adaptation_module_loss /= (num_updates * self.cfg.num_adaptation_module_substeps)
      
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss,mean_entropy_loss
