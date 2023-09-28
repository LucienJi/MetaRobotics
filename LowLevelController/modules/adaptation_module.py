import torch.nn as nn
import torch
from utils.torch_utils import  get_activation,check_cnnoutput,split_and_pad_trajectories,unpad_trajectories

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
        input = torch.cat([obs, privileged_obs], dim = -1)
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


class RNNHisotryEncoder(nn.Module):
    """固定历史长度的RNN编码器"""
    def __init__(self, 
                 num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 rnn_type='lstm',
                 rnn_hidden_size=256,
                 adaptation_module_branch_hidden_dims = [256, 128]):
        super(RNNHisotryEncoder, self).__init__()
        activation_fn = get_activation(activation)
        self.feature_extractor = nn.Linear(num_obs, rnn_hidden_size)
        self.memory = Memory(input_size=rnn_hidden_size, type=rnn_type, hidden_size=rnn_hidden_size)

        #! hidden size = rnn_hidden_size 
        estimator_branch = []
        estimator_branch.append(nn.Linear(rnn_hidden_size, adaptation_module_branch_hidden_dims[0]))
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                estimator_branch.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], num_latent))
            else:
                estimator_branch.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                estimator_branch.append(activation_fn)

        self.estimator = nn.Sequential(*estimator_branch)
    
    def reset(self,dones = None):
        self.memory.reset(dones)
    def forward(self,obs,hidden_states = None):
        pre_input = self.feature_extractor(obs)
        h = self.memory(pre_input,  hidden_states)[:,-1,:]
        latent_state = self.estimator(h)
        return h, latent_state #! (shape = (bz,time_step, num_latent))

    

class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256,batch_first = True):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=batch_first)
        self.hidden_states = None
    
    def forward(self, input,hidden_states=None):
        #! input.shape = (bz, T, extra_dim) or (bz, extradim)
        bz = input.shape[0]
        self.hidden_states = hidden_states if hidden_states is not None else self.hidden_states 
        if self.hidden_states is None or self.hidden_states[0].shape[1] != bz:
            print("Initializing hidden states")
            self.init_hidden_states(bz)
        if len(input.shape) == 2:
            input = input.unsqueeze(1)
        out, self.hidden_states = self.rnn(input, self.hidden_states)
        return out
    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
    def init_hidden_states(self, batch_size):
        if isinstance(self.rnn, nn.LSTM):
            self.hidden_states = (torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size),
                                  torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        else:
            self.hidden_states = (torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
    
    def get_hidden_states(self,obs = None):
        if (self.hidden_states is None) and( obs is not None):
            raise ValueError("Hidden states is None, please call forward() first")
        elif self.hidden_states:
            self.init_hidden_states(obs.shape[0])
        return self.hidden_states