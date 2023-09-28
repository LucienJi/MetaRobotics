import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation, MultivariateGaussianDiagonalCovariance, init_orhtogonal
from .adaptation_module import RNNHisotryEncoder,Memory,MLPExpertEncoder

class RNNActor(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_latent,
                 num_actions,
                 activation = 'elu',
                 rnn_hidden_size = 256,
                 actor_hidden_dims = [512,256,128]):
        super().__init__()
        mlp_input_dim_a = num_obs + num_latent  + rnn_hidden_size #! (bz,T,obs_dim + latent_dim)
        actor_layers = []
        activation_fn = get_activation(activation)
        actor_layers.extend(
            [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]),activation_fn]
            )
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l],num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l],actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor_body = nn.Sequential(*actor_layers)

    def forward(self,obs,latent, hidden_states):
        """
        obs.shape = (bz, obs_dim), latent.shape (bz,latent_dim), hidden_states.shape = (bz, rnn_hidden_size)
        """
        input = torch.cat([obs, latent,hidden_states], dim = -1)
        output = self.actor_body(output) #! (bz,T,num_actions)
        return output



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
        input = torch.cat([obs, latent_dim], dim = -1)
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
        input = torch.cat([obs, privileged_obs], dim = -1)
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
                 rnn_type = 'lstm',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [512, 256, 128],
                 rnn_hidden_size = 256,
                 init_noise_std = 1.0):
        super().__init__()

        # Adaptation Module
        self.adaptation_module = RNNHisotryEncoder(
            num_obs = num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_history = num_history,
            num_latent=num_latent,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            adaptation_module_branch_hidden_dims = adaptation_module_branch_hidden_dims
        )

        # Actor Module
        self.actor = RNNActor(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_latent=num_latent,
            num_actions=num_actions,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
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

    def reset(self, dones=None):
        self.actor.reset(dones)
        self.adaptation_module.reset(dones)

    def act_student(self, obs, privileged_obs, obs_history,
                    masks=None,hidden_states_e = None ,hidden_states_a=None):
        # obs_dict: obs, obs_history, privileged_obs
        latent = self.adaptation_module.forward(obs_history, hidden_states_e) #(bz, T, latent_dim)
        action_mean = self.actor.forward(obs_history, latent,hidden_states_a) # (bz,a)
        self.distribution.update(action_mean)
        return self.distribution.sample()

    def act_expert(self, obs, privileged_obs, obs_history,
                   masks=None,hidden_states_e = None ,hidden_states_a=None):
        # obs_dict: obs, obs_history, privileged_obs
        latent = self.expert_encoder.forward(obs, privileged_obs) # (bz, dim)
        action_mean = self.actor.forward(obs, latent,hidden_states_a) # (bz,a)
        self.distribution.update(action_mean)
        return self.distribution.sample()
    
    def get_hidden_states(self):
        return self.actor.memory_a.hidden_states,self.adaptation_module.memory.hidden_states


    def get_student_latent(self,obs, privileged_obs, obs_history,
                    masks=None,hidden_states_e = None ,):
        return self.adaptation_module.forward(obs_history, hidden_states_e)[:,-1,:] #(bz, T, latent_dim)
    def get_expert_latent(self,obs,privileged_obs):
        return self.expert_encoder.forward(obs, privileged_obs) # (bz, latent_dim)

    def get_actions_log_prob(self,actions):
        return self.distribution.get_actions_log_prob(actions)

    def act_inference(self,obs_dict,batch_mode = False):
        if not batch_mode:
            latent = self.adaptation_module.forward(obs_dict['obs']) #(bz, 1, latent_dim)
            latent = latent[:,-1,:]
            return self.actor.forward(obs_dict['obs'], latent)
        else:
            latent = self.adaptation_module.forward(obs_dict['obs_history'])
            return self.actor.forward(obs_dict['obs_history'], latent)

    def evaluate(self,obs,privileged_observations):
        value = self.critic.forward(obs, privileged_observations)
        return value 
        
