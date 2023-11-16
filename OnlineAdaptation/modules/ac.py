import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation,init_orhtogonal,SquashedGaussian
from .adaptation_module import body_index,hip_index,thigh_index,calf_index,edge_index,GraphEncoder,mlp , MLPHistoryEncoder
from torch_geometric.nn import ResGatedGraphConv, GatedGraphConv
from .forward_model import GraphForward,MLPForward
from .vq_torch.vq_quantize import VectorQuantize


class VQActorCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_history,
                 num_latent,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0,
                 use_forward = False):
        super().__init__()

        # Actor 
        self.use_forward = use_forward
        codebook_size = 128 # num code in 1 head 
        heads,codebook_dim = 8 , 32 
        dim = heads * codebook_dim
        input_size = num_obs_history
        act_fn = get_activation(activation)
        self.adaptation_module = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            act_fn,
            nn.Linear(input_size * 2, dim),
        )
        self.vq = VectorQuantize(dim = dim , codebook_size=codebook_size,codebook_dim=codebook_dim,
                           heads=heads, separate_codebook_per_head=True,
                           orthogonal_reg_weight=0.0,
                           )
        
        actor_input_size = num_obs + dim
        self.actor = mlp(
            input_dim=actor_input_size,
            out_dim=num_actions,hidden_sizes=actor_hidden_dims,activations=get_activation(activation)
        )

        # Critic 
        critic_input_size = num_obs + num_privileged_obs
        self.critic = mlp(
            input_dim=critic_input_size,
            out_dim=1, hidden_sizes=critic_hidden_dims,activations=get_activation(activation)
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
    def update_distribution(self, obs, observation_history):
        observation_history = observation_history.reshape(observation_history.shape[0],-1)
        latent= self.adaptation_module(observation_history)
        latent,indices, loss  = self.vq(latent)
        mean = self.actor(torch.cat([obs, latent], dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self,obs, observation_history, **kwargs):
        self.update_distribution(obs,observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, ob, policy_info={}):
        self.update_distribution(ob["obs"], ob["obs_history"])
        return self.distribution.mean

    def evaluate(self, obs, obs_history, privileged_observations, **kwargs):
        value = self.critic(torch.cat([obs, privileged_observations], dim=-1))
        return value

    
    
    # region
    #--------------------- Training ---------------------#
    def get_latent_and_loss(self, observation_history):
        observation_history = observation_history.reshape(observation_history.shape[0],-1)
        latent = self.adaptation_module(observation_history)
        latent,indices, loss = self.vq(latent)
        return latent, loss 
    
    def _update_with_latent(self,obs, latent):
        mean = self.actor(torch.cat([obs, latent], dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)
    # endregion



class ActorCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_history,
                 num_latent,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0,
                 use_forward = True):
        super().__init__()
        self.use_forward = use_forward
        # Adaptation module
        self.adaptation_module = MLPHistoryEncoder(num_obs,
                                                   num_privileged_obs,
                                                   num_obs_history,
                                                   num_history,
                                                   num_latent,
                                                   activation)
        
        # Actor 
        mlp_input_size = num_obs + num_latent
        self.actor = mlp(mlp_input_size, num_actions, actor_hidden_dims, get_activation(activation))
        
        # Critic
        mlp_critic_input = num_privileged_obs + num_obs 
        self.critic = mlp(mlp_critic_input, 1, critic_hidden_dims, get_activation(activation))

        # Forward 
        if self.use_forward:
            self.forward_model = MLPForward(num_obs,num_latent,num_actions,activation)
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

    
    def update_distribution(self, obs, observation_history):
        node_latent = self.adaptation_module(observation_history)
        mean = self.actor(obs, node_latent)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self,obs, observation_history, **kwargs):
        self.update_distribution(obs,observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, ob, policy_info={}):
        self.update_distribution(ob["obs"], ob["obs_history"])
        return self.distribution.mean

    def evaluate(self, obs, obs_history, privileged_observations, **kwargs):
        value = self.critic(torch.cat([obs, privileged_observations], dim=-1))
        return value

    
    
    # region
    #--------------------- Training ---------------------#
    def get_latent(self, observation_history):
        return self.adaptation_module(observation_history)
    
    def _update_with_latent(self,obs, latent):
        mean = self.actor(obs, latent)
        self.distribution = Normal(mean, mean * 0. + self.std)
    # endregion
        
    
class GraphActor(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_latent,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128]):
        super().__init__()
        self.num_latent = num_latent
        # graph info
        node_base = torch.tensor(list(body_index.values()),dtype=torch.long).squeeze()
        node_hip = torch.stack([torch.tensor(list(hip_index.values()),dtype=torch.long)],dim=0).squeeze()
        node_thigh = torch.stack([torch.tensor(list(thigh_index.values()),dtype=torch.long)],dim=0).squeeze()
        node_calf = torch.stack([torch.tensor(list(calf_index.values()),dtype=torch.long)],dim=0).squeeze()
        activation_fn = get_activation(activation)
        self.node_base = nn.Parameter(node_base, requires_grad=False)
        self.node_hip = nn.Parameter(node_hip, requires_grad=False)
        self.node_thigh = nn.Parameter(node_thigh, requires_grad=False)
        self.node_calf = nn.Parameter(node_calf, requires_grad=False) 
        self.edge = nn.Parameter(torch.as_tensor(edge_index, dtype=torch.long).contiguous().t(),requires_grad=False)
        # Pipeline
        # obs-> node, concat with latent -> node latent
        # node latent -> node level policy -> node action
        base_input_size =  len(list(body_index.values())[0])
        hip_input_size = len(list(hip_index.values())[0])
        thigh_input_size =len(list(thigh_index.values())[0])
        calf_input_size = len(list(calf_index.values())[0])
        self.base_net = mlp(base_input_size, num_latent, [128], activation_fn)
        self.hip_net = mlp(hip_input_size, num_latent, [128], activation_fn)
        self.thigh_net = mlp(thigh_input_size, num_latent, [128], activation_fn)
        self.calf_net = mlp(calf_input_size, num_latent, [128], activation_fn)

        # graph neural network 
        self.gn = ResGatedGraphConv(in_channels=2 * num_latent,
                                    out_channels=2 * num_latent)
        self.act = activation_fn
        self.gn2 = ResGatedGraphConv(in_channels=2 * num_latent,
                                    out_channels=num_latent)
        
        # graph policy : (base, hip, thigh, calf) -> leg_control 
        self.leg_policy = mlp(4*num_latent, 3, [256,128], activation_fn)
        self.FL_Leg = nn.Parameter(torch.tensor([0,1,5,9],dtype=torch.long), requires_grad=False)
        self.FR_Leg = nn.Parameter(torch.tensor([0,2,6,10],dtype=torch.long), requires_grad=False)
        self.RL_Leg = nn.Parameter(torch.tensor([0,3,7,11],dtype=torch.long), requires_grad=False)
        self.RR_Leg = nn.Parameter(torch.tensor([0,4,8,12],dtype=torch.long), requires_grad=False)

    
    def _obs2node(self, obs):
        # obs.shape = (bz, n_obs)
        base = obs[:,self.node_base].unsqueeze(1) # (bz, 1, n_base)
        hip = obs[:,self.node_hip]# (bz, 4, 4)
        thigh = obs[:,self.node_thigh]
        calf = obs[:,self.node_calf]
        base = self.base_net(base)
        hip = self.hip_net(hip)
        thigh = self.thigh_net(thigh)
        calf = self.calf_net(calf)
        return torch.cat([base,hip,thigh,calf],dim=1) # (bz, n_node,num_latent)
    def forward(self, obs, latent):
        # obs.shape = (bz, n_obs)
        # latent.shape = (bz,n_node, num_latent)
        obs_nodes = self._obs2node(obs) # (bz, n_node, num_latent) 
        nodes_latent = torch.cat([obs_nodes,latent],dim=-1) # (bz, n_node, 2*num_latent)
        nodes_latent = self.gn(nodes_latent, self.edge) # (bz, n_node, 2*num_latent)
        nodes_latent = self.act(nodes_latent)
        nodes_latent = self.gn2(nodes_latent, self.edge) 
        FL_Leg_latent = nodes_latent[:,self.FL_Leg,:].reshape(-1,4*self.num_latent)
        FR_Leg_latent = nodes_latent[:,self.FR_Leg,:].reshape(-1,4*self.num_latent)
        RL_Leg_latent = nodes_latent[:,self.RL_Leg,:].reshape(-1,4*self.num_latent)
        RR_Leg_latent = nodes_latent[:,self.RR_Leg,:].reshape(-1,4*self.num_latent)
        FL_Leg_action = self.leg_policy(FL_Leg_latent)
        FR_Leg_action = self.leg_policy(FR_Leg_latent)
        RL_Leg_action = self.leg_policy(RL_Leg_latent)
        RR_Leg_action = self.leg_policy(RR_Leg_latent)
        return torch.cat([FL_Leg_action,FR_Leg_action,RL_Leg_action,RR_Leg_action],dim=1) # (bz, 12)


class GraphActorCritic(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_history,
                 num_latent,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 init_noise_std = 1.0,
                 use_forward = True):
        super().__init__()
        self.use_forward = use_forward
        self.adaptation_module = GraphEncoder(num_obs, 
                                              num_history, 
                                              num_latent,
                                              activation)    
        
        # Actor 
        self.actor = GraphActor(num_obs,
                                      num_latent,
                                      num_actions,
                                      activation,
                                      actor_hidden_dims)
        
        # Critic
        mlp_critic_input = num_privileged_obs + num_obs 
        self.critic = mlp(mlp_critic_input, 1, critic_hidden_dims, get_activation(activation))

        # Forward 
        if self.use_forward:
            self.forward_model = GraphForward(num_obs,num_latent,num_actions,activation)
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

    
    def update_distribution(self, obs, observation_history):
        node_latent = self.adaptation_module(observation_history)
        mean = self.actor(obs, node_latent)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self,obs, observation_history, **kwargs):
        self.update_distribution(obs,observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, ob, policy_info={}):
        self.update_distribution(ob["obs"], ob["obs_history"])
        return self.distribution.mean

    def evaluate(self, obs, obs_history, privileged_observations, **kwargs):
        value = self.critic(torch.cat([obs, privileged_observations], dim=-1))
        return value

    
    
    # region
    #--------------------- Training ---------------------#
    def get_latent(self, observation_history):
        return self.adaptation_module(observation_history)
    
    def _update_with_latent(self,obs, latent):
        mean = self.actor(obs, latent)
        self.distribution = Normal(mean, mean * 0. + self.std)
    # endregion