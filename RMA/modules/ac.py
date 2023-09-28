import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation, MultivariateGaussianDiagonalCovariance, init_orhtogonal
from .adaptation_module import TCNHistoryEncoder, MLPExpertEncoder

class BaseActor(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_latent,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],):
        super().__init__()
        mlp_input_dim_a = num_obs + num_latent  
        actor_layers = []
        activation_fn = get_activation(activation)
        actor_layers.extend(
            [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]),
            activation_fn]
            )   
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l],num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l],actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor_body = nn.Sequential(*actor_layers)

    def forward(self,obs,latent_dim):
        """
        obs.shape = (bz, obs_dim)
        """
        input = torch.cat([obs, latent_dim], dim = 1)
        output = self.actor_body(input)
        return output

class BaseCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_latent,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],):
        super().__init__()
        mlp_input_dim_c = num_obs + num_privileged_obs
        critic_layers = []
        activation_fn = get_activation(activation)
        critic_layers.extend([
            nn.Linear(mlp_input_dim_c, actor_hidden_dims[0]),
            activation_fn]
            )
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                critic_layers.append(nn.Linear(actor_hidden_dims[l],1))
            else:
                critic_layers.append(nn.Linear(actor_hidden_dims[l],actor_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic_body = nn.Sequential(*critic_layers)
    
    def forward(self,obs,privileged_obs):
        """
        obs.shape = (bz, obs_dim)
        privileged_obs.shape = (bz, privileged_obs_dim)
        """
        input = torch.cat([obs, privileged_obs], dim = 1)
        output = self.critic_body(input)
        return output

class ActorCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 num_latent,
                 num_history,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0):
        super().__init__()

        # Expert Module 
        self.expert_encoder = MLPExpertEncoder(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_latent=num_latent,
            activation=activation,
            adaptation_module_branch_hidden_dims=adaptation_module_branch_hidden_dims,
        )

        # Adaptation Module
        self.adaptation_module = TCNHistoryEncoder(
            num_obs = num_obs,
            num_history = num_history,
            num_latent=num_latent,
            activation=activation,
        )

        # Actor Module
        self.actor = BaseActor(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_latent=num_latent,
            num_actions=num_actions,
            activation=activation,
            actor_hidden_dims=actor_hidden_dims,
        )
        self.distribution = MultivariateGaussianDiagonalCovariance(
            dim = num_actions,
            init_std=init_noise_std,
        )
        
        # Critic Module

        self.critic = BaseCritic(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_latent=num_latent,
            num_actions=num_actions,
            activation=activation,
            actor_hidden_dims=critic_hidden_dims,
        )
        Normal.set_default_validate_args = False

        self.expert_encoder.apply(init_orhtogonal)
        self.adaptation_module.apply(init_orhtogonal)
        self.actor.apply(init_orhtogonal)
        self.critic.apply(init_orhtogonal)
        
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    
    def act_student(self, obs, privileged_obs, obs_history):
        # obs_dict: obs, obs_history, privileged_obs
        latent = self.adaptation_module.forward(obs_history)
        action_mean = self.actor.forward(obs, latent)
        self.distribution.update(action_mean)
        return self.distribution.sample()

    def act_expert(self, obs, privileged_obs, obs_history):
        # obs_dict: obs, obs_history, privileged_obs
        latent = self.expert_encoder.forward(obs, privileged_obs)
        action_mean = self.actor.forward(obs, latent)
        self.distribution.update(action_mean)
        return self.distribution.sample()
    
    def get_student_latent(self,obs_history):
        return self.adaptation_module.forward(obs_history)
    def get_expert_latent(self,obs,privileged_obs):
        return self.expert_encoder.forward(obs, privileged_obs)


    def get_actions_log_prob(self,actions):
        return self.distribution.get_actions_log_prob(actions)

    def act_inference(self,obs_dict):
        latent = self.adaptation_module.forward(obs_dict['obs_history'])
        return self.actor.forward(obs_dict['obs'], latent)

    def evaluate(self,obs,privileged_observations):
        value = self.critic.forward(obs, privileged_observations)
        return value 
        
