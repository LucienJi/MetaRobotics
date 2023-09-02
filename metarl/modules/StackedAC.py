import torch
import torch.nn as nn
from torch.distributions import Normal
MIN_STD = 0.0001



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


        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

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

    def update_distribution(self, observation_history):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
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
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)


from .utils import init_zero,freeze,unfreeze,MLP
import copy 

class ControlActorCritic(nn.Module):
    def __init__(self, 
                 expert_model:ActorCritic,
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_reward_dim,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0):
        super().__init__()
        if expert_model is None:
            self.model = ActorCritic(
                    num_obs,
                    num_privileged_obs,
                    num_obs_history,
                    num_actions,
                    activation,
                    actor_hidden_dims,
                    critic_hidden_dims,
                    adaptation_module_branch_hidden_dims,
                    init_noise_std)
        else:
            self.model = expert_model 
        freeze(self.model)
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        activation = get_activation(activation)

        # -------- Control Actor -------- #
        actor_input_dim = num_obs_history + num_reward_dim
        self.actor_zero_layer1 = nn.Linear(
            in_features=actor_input_dim,
            out_features=self.model.actor_obs_feature,
            bias=True
        )
        self.actor_trainable_block = copy.deepcopy(self.model.actor_blocks)
        self.actor_zero_layer2 = nn.Linear(
            in_features=self.model.actor_action_feature + num_reward_dim,
            out_features=num_actions,
            bias=True
        )
        unfreeze(self.actor_trainable_block)
        self.actor_zero_layer1.apply(init_zero)
        self.actor_zero_layer2.apply(init_zero)

        # -------- Control Critic -------- #
        critic_input_dim = num_obs_history + num_reward_dim
        self.critic_zero_layer1 = nn.Linear(
            in_features=critic_input_dim,
            out_features=self.model.critic_obs_feature,
            bias=True
        )
        self.critic_trainable_block = copy.deepcopy(self.model.critic_blocks)
        self.critic_zero_layer2 = nn.Linear(
            in_features=self.model.critic_value_feature + num_reward_dim,
            out_features= 1,
            bias=True
        )
        unfreeze(self.critic_trainable_block)
        self.critic_zero_layer1.apply(init_zero)
        self.critic_zero_layer2.apply(init_zero)

        self.distribution = None

        self.max_logstd = 1.0
        self.logstd = nn.Sequential(
            nn.Linear(in_features=actor_input_dim,out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=num_actions)
        )
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def _actor_default_forward(self,observation_history,reward_scales):
        latent = self.model.adaptation_module.forward(observation_history)
        raw_actor_feature = self.model.actor_encoder(torch.cat([observation_history,latent],dim=-1))
        
        delta_actor_feature = self.actor_zero_layer1(torch.cat([observation_history,reward_scales],dim=-1))
        actor_feature = raw_actor_feature + delta_actor_feature

        raw_actor_action_feature = self.model.actor_blocks(raw_actor_feature)
        raw_actor_action_mean = self.model.actor_output_layer(raw_actor_action_feature)

        delta_actor_action_feature = self.actor_trainable_block(actor_feature)
        delta_actor_action_feature = torch.cat([delta_actor_action_feature,reward_scales],dim=-1)
        delta_actor_action_mean = self.actor_zero_layer2(delta_actor_action_feature)
        actions_mean = raw_actor_action_mean + delta_actor_action_mean
        return actions_mean
    def _actor_std_default_forward(self,observation_history,reward_scales):
        logstd = self.logstd.forward(torch.cat([observation_history,reward_scales],dim =-1))
        actions_logstd = torch.clamp(logstd,max = self.max_logstd)
        actions_std = actions_logstd.exp()
        return actions_std

    def _critc_default_forward(self,observation_history,reward_scales):
        latent = self.model.adaptation_module.forward(observation_history)
        raw_critic_feature = self.model.critic_encoder(torch.cat([observation_history,latent],dim=-1))
        delta_critic_feature = self.critic_zero_layer1(torch.cat([observation_history,reward_scales],dim=-1))
        critic_feature = raw_critic_feature + delta_critic_feature

        raw_critic_value_feature = self.model.critic_blocks(raw_critic_feature)
        raw_critic_values = self.model.critic_output_layer(raw_critic_value_feature)

        delta_critic_value_feature = self.critic_trainable_block(critic_feature)
        delta_critic_value_feature = torch.cat([delta_critic_value_feature,reward_scales],dim=-1)
        delta_critic_values = self.critic_zero_layer2(delta_critic_value_feature)
        
        values = raw_critic_values + delta_critic_values
        return values
    def update_distribution(self, observation_history,reward_scales):
        mean = self._actor_default_forward(observation_history,reward_scales)
        std = self._actor_std_default_forward(observation_history,reward_scales)
        self.distribution = Normal(mean, mean*0. + torch.clamp(std,min = MIN_STD))

    def act(self, observation_history,reward_scales, **kwargs):
        self.update_distribution(observation_history,reward_scales)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_student(self, observation_history,reward_scales, policy_info={}):
        latent = self.model.adaptation_module(observation_history)
        actions_mean = self._actor_default_forward(observation_history,reward_scales)
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean
    
    def act_inference(self,obs_dict,reward_scales):
        return self.act_student(obs_dict["obs_history"],reward_scales)

    def act_teacher(self, observation_history, privileged_info,reward_scales, policy_info={}):
        latent = privileged_info
        raw_actor_feature = self.model.actor_encoder(torch.cat([observation_history,latent],dim=-1))
        delta_actor_feature = self.actor_zero_layer1(torch.cat([observation_history,reward_scales],dim=-1))
        actor_feature = raw_actor_feature + delta_actor_feature

        raw_actor_action_feature = self.model.actor_blocks(raw_actor_feature)
        raw_actor_action_mean = self.model.actor_output_layer(raw_actor_action_feature)

        delta_actor_action_feature = self.actor_trainable_block(actor_feature)
        delta_actor_action_feature = torch.cat([delta_actor_action_feature,reward_scales],dim=-1)
        delta_actor_action_mean = self.actor_zero_layer2(delta_actor_action_feature)
        actions_mean = raw_actor_action_mean + delta_actor_action_mean
        policy_info["latents"] = privileged_info
        return actions_mean
    def evaluate(self, observation_history, privileged_observations,reward_scales, **kwargs):
        value = self._critc_default_forward(observation_history,reward_scales)
        return value

class MOActorCritic(nn.Module):
    def __init__(self,num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_reward_dim,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0) -> None:
        super(MOActorCritic,self).__init__()
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
        mlp_input_dim_a = self.num_privileged_obs + self.num_obs_history + num_reward_dim
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
        mlp_input_dim_c = self.num_privileged_obs + self.num_obs_history + num_reward_dim
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

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

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

    def update_distribution(self, observation_history,reward_scales):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent,reward_scales), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, reward_scales, **kwargs):
        self.update_distribution(observation_history,reward_scales)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, reward_scales,policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"],reward_scales,policy_info)

    def act_inference(self, ob,reward_scales, policy_info={}):
        return self.act_student(ob["obs_history"],reward_scales, policy_info=policy_info)

    def act_student(self, observation_history,reward_scales, policy_info={}):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observation_history, latent,reward_scales), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info,reward_scales, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info,reward_scales), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations,reward_scales, **kwargs):
        value = self.critic_body(torch.cat((observation_history, privileged_observations,reward_scales), dim=-1))
        return value

    def get_student_latent(self, observation_history,reward_scales = None ):
        return self.adaptation_module(observation_history)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
