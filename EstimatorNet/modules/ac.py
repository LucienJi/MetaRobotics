import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation, MultivariateGaussianDiagonalCovariance, init_orhtogonal
from .state_estimator import EstimatorNet

class BaseActor(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_latent, #! 应该是 这是 latent + 3 + 4 + 4 (vel, foot_height, contact)
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
                 num_privileged_obs, #! 包含 vel, foot_height, contact
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
                 estimator_hidden_dims = [512, 256, 128],
                 init_noise_std = 1.0):
        super().__init__()

        # Expert Module 
        self.estimator = EstimatorNet(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_history=num_history,
            num_latent=num_latent,
            activation=activation,
            estimator_branch=estimator_hidden_dims
        )
            

        # Actor Module
        self.actor = BaseActor(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_latent=num_latent +  3 + 4 + 4 , #! 
            num_actions=num_actions,
            activation=activation,
            actor_hidden_dims=actor_hidden_dims,
        )
        self.distribution = None
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
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.estimator.apply(init_orhtogonal)
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

    #! rollout 的时候需要随机性, 这里是 bootstrap
    def act(self, obs, privileged_obs, obs_history):
        # obs_dict: obs, obs_history, privileged_obs
        vel, foot_height, contact , latent = self.estimator.forward(obs_history)
        latent = torch.cat([vel,foot_height,contact,latent],dim = 1)
        action_mean = self.actor.forward(obs, latent)
        self.distribution = Normal(action_mean, action_mean * 0. + self.std)
        return self.distribution.sample()


    def get_actions_log_prob(self,actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self,obs_dict):
        vel, foot_height, contact,latent = self.estimator.forward(obs_dict['obs_history'])
        latent = torch.cat([vel, foot_height, contact,latent],dim = 1)
        return self.actor.forward(obs_dict['obs'], latent)

    def evaluate(self,obs,privileged_observations):
        value = self.critic.forward(obs, privileged_observations)
        return value 
    
    # region 
    # ------------ Training ------------ 

    def _update_with_latent(self,obs, latent):
        mean = self.actor(torch.cat([obs, latent], dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    # endregion
