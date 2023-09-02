import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import get_activation,ortho_init
MIN_STD = 0.0001


"""
Actor-Critic module
implement the actor-critic module with adaptation module
"""

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 latent_dim = 64,
                 init_noise_std = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(activation)

        # EnvEncoder
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],latent_dim)) #! latent_dim
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)


        ####################
        # Policy
        mlp_input_dim_a = self.latent_dim + self.num_obs 
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

        ####################
        # Value function
        mlp_input_dim_c = self.num_privileged_obs + self.num_obs
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


        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        self.apply(ortho_init)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observation_history,obs):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((obs, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, obs):
        self.update_distribution(observation_history,obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    
    def act_inference(self, ob, policy_info={}):
        self.update_distribution(ob['obs_history'],ob['obs'])
        return self.distribution.mean

    def evaluate(self, obs, privileged_observations, **kwargs):
        value = self.critic_body(torch.cat((obs, privileged_observations), dim=-1))
        return value

    def get_latent(self, observation_history):
        return self.adaptation_module(observation_history)

