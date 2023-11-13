import torch.nn as nn
import torch
from utils.torch_utils import  get_activation,check_cnnoutput
"""
obs_buf: 
    base_vel, 3 : 0:3
    base_ang, 3 : 3:6
    project_gravity, 3: 6:9
    cmd, 3 : 9:12
    dof_pos, 12 : 12:24
    dof_vel, 12 : 24:36
    actions, 12 : 36:48
    contact, 4  : 48:52
"""
body_index = {
    '0': [0,1,2,3,4,5,6,7,8,9,10,11,48,49,50,51],
}

hip_index = {
    '1': [12,24,36,48],
    '2': [15,27,39,49],
    '3': [18,30,42,50],
    '4': [21,33,45,51],
}
thigh_index = {
    '5': [13,25,37,48],
    '6': [16,28,40,49],
    '7': [19,31,43,50],
    '8': [22,34,46,51],
}
calf_index = {
    '9': [14,26,38,48],
    '10': [17,29,41,49],
    '11': [20,32,44,50],
    '12': [23,35,47,51],
}

edge_index = [
    [0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],[0,10],[0,11],[0,12],
    [1,5],[5,9],
    [2,6],[6,10],
    [3,7],[7,11],
    [4,8],[8,12],
]

class GraphEncoder(nn.Module):
    def __init__(self,
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',):
        super(GraphEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_latent = num_latent
        activation_fn = get_activation(activation)
        # graph info
        self.nodes = nn.ParameterDict({k:nn.Parameter(torch.as_tensor(v,dtype=torch.long),requires_grad=False) for k,v in node_info.items()})
        self.n_node = len(self.nodes.values())
        output_size = num_latent
    
        


class MLPExpertEncoder(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_latent,
                 activation = 'elu',
                 adaptation_module_branch_hidden_dims = [256, 128],):
        super(MLPExpertEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_latent = num_latent
        activation_fn = get_activation(activation)
        input_size = num_obs + num_privileged_obs
        output_size = num_latent
        expert_encoder_layers = []
        expert_encoder_layers.append(nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]))
        expert_encoder_layers.append(activation_fn)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                expert_encoder_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],output_size))
            else:
                expert_encoder_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                expert_encoder_layers.append(activation_fn)
        self.encoder = nn.Sequential(*expert_encoder_layers)

    def forward(self,obs,privileged_obs):
        """
        obs.shape = (bz, obs_dim)
        privileged_obs.shape = (bz, privileged_obs_dim)
        """
        bs = obs.shape[0]
        input = torch.cat([obs, privileged_obs], dim = 1)
        output = self.encoder(input)
        return output


class TCNHistoryEncoder(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',):
        super(TCNHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history  
        self.num_latent = num_latent    

        activation_fn = get_activation(activation)
        self.tsteps = tsteps = num_history
        input_size = num_obs
        output_size = num_latent
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            activation_fn,
            nn.Linear(128, 32),
        )
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            last_dim = 32 * 3
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            last_dim = 32 * 3
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            last_dim = 32 * 3
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            last_dim = check_cnnoutput(input_size = (tsteps,32), list_modules = [self.conv_layers])

        self.linear_output = nn.Sequential(
            nn.Linear(last_dim, output_size)
            )

    def forward(self, obs_history):
        """
        obs_history.shape = (bz, T , obs_dim)
        """
        bs = obs_history.shape[0]
        T = self.tsteps
        projection = self.encoder(obs_history) # (bz, T , 32) -> (bz, 32, T) bz, channel_dim, Temporal_dim
        output = self.conv_layers(projection.permute(0, 2, 1)) # (bz, last_dim)
        output = self.linear_output(output)
        return output
    

class MLPHistoryEncoder(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 adaptation_module_branch_hidden_dims = [256, 128],):
        super(MLPHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent

        input_size = num_obs * num_history
        output_size = num_latent

        activation = get_activation(activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], output_size))
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.encoder = nn.Sequential(*adaptation_module_layers)
    def forward(self, obs_history):
        """
        obs_history.shape = (bz, T , obs_dim)
        """
        bs = obs_history.shape[0]
        T = self.num_history
        output = self.encoder(obs_history.reshape(bs, -1))
        return output

