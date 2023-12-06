import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation,init_orhtogonal
from .adaptation_module import mlp, MLPExpertEncoder,MLPHistoryEncoder

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
        self.adaptation_module = MLPHistoryEncoder(
            num_obs = num_obs,
            num_privileged_obs= num_privileged_obs,
            num_obs_history=num_obs_history,
            num_history=num_history,
            num_latent=num_latent,
            activation=activation,
            adaptation_module_branch_hidden_dims=adaptation_module_branch_hidden_dims,
        )
        # Actor Module
        actor_input_size = num_obs + num_latent
        actor_act = get_activation(activation)
        self.actor = mlp(
            input_dim=actor_input_size,
            out_dim=num_actions,hidden_sizes=actor_hidden_dims,activations=actor_act
        )

        # Critic 
        critic_input_size = num_obs + num_privileged_obs
        self.critic = mlp(
            input_dim=critic_input_size,
            out_dim=1, hidden_sizes=critic_hidden_dims,activations=actor_act
        )
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        self.apply(init_orhtogonal)
        
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
        action_mean = self.actor.forward(torch.cat([obs, latent], dim=-1))
        self.distribution = Normal(action_mean, action_mean * 0. + self.std)
        return self.distribution.sample()

    def act_expert(self, obs, privileged_obs, obs_history):
        # obs_dict: obs, obs_history, privileged_obs
        latent = self.expert_encoder.forward(obs, privileged_obs)
        action_mean = self.actor.forward(torch.cat([obs, latent], dim=-1))
        self.distribution = Normal(action_mean, action_mean * 0. + self.std)
        return self.distribution.sample()
    
    def get_actions_log_prob(self,actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self,obs_dict):
        latent = self.adaptation_module.forward(obs_dict['obs_history'])
        obs = obs_dict['obs']
        return self.actor.forward(torch.cat([obs, latent], dim=-1))

    def evaluate(self,obs,privileged_observations):
        value = self.critic.forward(torch.cat([obs, privileged_observations], dim=-1))
        return value 

    # region 
    # ----------- Training ---------------
    def get_student_latent(self,obs_history):
        return self.adaptation_module.forward(obs_history)
    def get_expert_latent(self,obs,privileged_obs):
        return self.expert_encoder.forward(obs, privileged_obs)
    def _update_with_latent(self,obs,latent):
        return self.actor.forward(obs, latent)
    # endregion
        
