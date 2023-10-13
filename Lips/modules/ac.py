import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation, MultivariateGaussianDiagonalCovariance, init_orhtogonal
from .lipsnet import LipsNet, ExpLipsNet    
from functorch import jacrev, vmap

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
        mlp_input_dim_a = num_obs   
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

    def forward(self,obs):
        """
        obs.shape = (bz, obs_dim)
        """
        input = obs
        output = self.actor_body(input)
        return output
    
    def check_jacob(self,x):
        with torch.no_grad():
            jacobi = vmap(jacrev(self.actor_body))(x)
        jac_norm = torch.norm(jacobi, 2, dim=(1,2)).unsqueeze(1)
        return jac_norm
    
class LipsActor(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_latent,
                 num_actions,
                 activation = 'lrelu',
                 actor_hidden_dims = [512, 256, 128],):
        super().__init__()
        mlp_input_dim_a = num_obs   
        activation_fn = get_activation(activation)
        identity_fn = get_activation('identity')
        k_act_fn = get_activation('tanh')

        self.lipnet = ExpLipsNet(
            f_sizes=[mlp_input_dim_a] + actor_hidden_dims + [num_actions],
            k_sizes=[mlp_input_dim_a] + actor_hidden_dims + [1],
            global_lips=False,
            k_init=25,
            k_max=50,
            f_hid_nonliear=activation_fn,
            f_out_nonliear=identity_fn,
            k_hid_act=k_act_fn,
            k_out_act=identity_fn,
            loss_lambda=0.001,
            eps=1e-4,
            squash_action=False,
        )

    def forward(self,obs):
        """
        obs.shape = (bz, obs_dim)
        """
        output = self.lipnet.lips_forward(obs)
        return output
    def raw_forward(self,obs):
        """
        Stage 1: nominal forward 
        """
        output = self.lipnet.raw_forward(obs)
        return output
    def sl_k_loss(self,obs):
        """
        obs.shape = (bz, obs_dim)
        """
        output = self.lipnet.sl_k_loss(obs)
        return output
    
    def reg_loss(self,obs):
        output = self.lipnet.l2_regularization(obs)
        return output

    def raw_jacob_check(self,obs,ord = 2):
        output = self.lipnet.raw_jacob_check(obs,ord)
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

class NorminalActorCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 num_latent,
                 num_history,
                 activation = 'relu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0):
        super().__init__()

        # Actor Module
        self.actor = BaseActor(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_latent=0,
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
            num_latent=0,
            num_actions=num_actions,
            activation=activation,
            actor_hidden_dims=critic_hidden_dims,
        )
        Normal.set_default_validate_args = False

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

    
    def act(self, obs, privileged_obs, obs_history):
        # obs_dict: obs, obs_history, privileged_obs
        action_mean = self.actor.forward(obs)
        self.distribution.update(action_mean)
        return self.distribution.sample()

    def get_actions_log_prob(self,actions):
        return self.distribution.get_actions_log_prob(actions)

    def act_inference(self,obs_dict):
        return self.actor.forward(obs_dict['obs'])

    def evaluate(self,obs,privileged_observations):
        value = self.critic.forward(obs, privileged_observations)
        return value 
        
class LipsActorCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 num_latent,
                 num_history,
                 activation = 'relu',
                 actor_hidden_dims = [256, 256],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 inference_stage = 1,
                 init_noise_std = 1.0):
        super().__init__()
        self.inference_stage = inference_stage
        # Actor Module
        self.actor = LipsActor(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_latent=0,
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
            num_latent=0,
            num_actions=num_actions,
            activation=activation,
            actor_hidden_dims=critic_hidden_dims,
        )
        Normal.set_default_validate_args = False
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
    
    def _raw_forward(self,obs):
        action_mean = self.actor.raw_forward(obs)
        self.distribution.update(action_mean)
        return self.distribution.sample()
    def _lips_forward(self,obs):
        action_mean = self.actor.forward(obs)
        self.distribution.update(action_mean)
        return self.distribution.sample()

    
    def act(self, obs, privileged_obs, obs_history,stage = 1):
        # obs_dict: obs, obs_history, privileged_obs
        if stage == 1:
            # raw
            return self._raw_forward(obs)
        elif stage == 3:
            # lips
            return self._lips_forward(obs)
        else:
            return self._raw_forward(obs)


    def get_actions_log_prob(self,actions):
        return self.distribution.get_actions_log_prob(actions)

    def act_inference(self,obs_dict):
        if self.inference_stage == 3:
            return self.actor.forward(obs_dict['obs'])
        else:
            return self.actor.raw_forward(obs_dict['obs'])

    def evaluate(self,obs,privileged_observations):
        value = self.critic.forward(obs, privileged_observations)
        return value 
