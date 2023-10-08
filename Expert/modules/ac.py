import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation,init_orhtogonal

class ActorCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0):
        super().__init__()
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        activation = get_activation(activation)
        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)



        # Policy
        mlp_input_dim_a = self.num_privileged_obs + self.num_obs_history
        self.actor_obs_feature = actor_hidden_dims[0]

        actor_layers = []
        self.actor_encoder = nn.Sequential(
            nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]),
            activation
        )
        actor_layers.append(self.actor_encoder)
        actor_blocks = []
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                self.actor_action_feature = actor_hidden_dims[l]
                self.actor_output_layer = nn.Linear(actor_hidden_dims[l], num_actions)
            else:
                actor_blocks.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_blocks.append(activation)
        self.actor_blocks = nn.Sequential(*actor_blocks)
        actor_layers.append(self.actor_blocks)
        actor_layers.append(self.actor_output_layer)
        self.actor_body = nn.Sequential(*actor_layers)


        # Value function
        mlp_input_dim_c = self.num_privileged_obs + self.num_obs_history
        critic_layers = []
        self.critic_encoder = nn.Sequential(
            nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]),
            activation
        )
        self.critic_obs_feature = critic_hidden_dims[0]
        critic_layers.append(self.critic_encoder)
        critic_blocks = []
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                self.critic_value_feature = critic_hidden_dims[l]
                self.critic_output_layer = nn.Linear(critic_hidden_dims[l], 1)
            else:
                critic_blocks.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_blocks.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        self.critic_blocks = nn.Sequential(*critic_blocks)
        critic_layers.append(self.critic_blocks)
        critic_layers.append(self.critic_output_layer)
        self.critic_body = nn.Sequential(*critic_layers)
        Normal.set_default_validate_args = False

        self.adaptation_module.apply(init_orhtogonal)
        self.actor_body.apply(init_orhtogonal)
        self.critic.apply(init_orhtogonal)

         # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    
    def update_distribution(self, observation_history):
        bz = observation_history.shape[0]
        latent = self.adaptation_module(observation_history.reshape(bz, -1))
        mean = self.actor_body(torch.cat((observation_history.reshape(bz, -1), latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        # print("act_student",self.adaptation_module)
        bz = observation_history.shape[0]
        latent = self.adaptation_module(observation_history.reshape(bz, -1))
        actions_mean = self.actor_body(torch.cat((observation_history.reshape(bz, -1), latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        bz = observation_history.shape[0]
        actions_mean = self.actor_body(torch.cat((observation_history.reshape(bz, -1), privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        bz = observation_history.shape[0]
        value = self.critic_body(torch.cat((observation_history.reshape(bz,-1), privileged_observations), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        bz = observation_history.shape[0]
        return self.adaptation_module(observation_history.reshape(bz,-1))
        
