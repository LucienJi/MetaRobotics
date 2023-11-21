import torch.nn as nn
import torch
from torch.distributions import Normal
from abc import ABC, abstractmethod
from typing import Tuple, Union
import json 
import numpy as np 

# minimal interface of the environment


def check_cnnoutput(input_size:list, list_modules):
    x = torch.randn(1, *input_size)
    for module in list_modules:
        x = module(x)
    return x.shape[1]

def init_zero(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight,val = 0.0)
        m.bias.data.fill_(0.0)


def init_orhtogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True



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
    elif act_name == "identity":
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None
    


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        Normal.set_default_validate_args = False
    
    def update(self,logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

    def sample(self):
        samples = self.distribution.sample()
        return samples

    def get_actions_log_prob(self,actions):
        actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        return actions_log_prob

    def entropy(self):
        return self.distribution.entropy()
    
    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
    @property
    def stddev(self):
        return self.distribution.stddev
    @property
    def mean(self):
        return self.distribution.mean
    



    

class MultivariateGaussianDiagonalCovariance2(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance2, self).__init__()
        assert(dim == 12)
        self.dim = dim
        self.std_param = nn.Parameter(init_std * torch.ones(dim // 2))
        self.distribution = None

    def update(self, logits):
    
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        self.distribution = Normal(logits, self.std.reshape(self.dim))

    def sample(self):
        samples = self.distribution.sample()
        return samples

    def get_actions_log_prob(self,actions):
        actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        return actions_log_prob

    def entropy(self):
        return self.distribution.entropy()
    
    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
    @property
    def stddev(self):
        return self.distribution.stddev
    @property
    def mean(self):
        return self.distribution.mean

MIN_LOG_NN_OUTPUT = -5
MAX_LOG_NN_OUTPUT = 2
SMALL_NUMBER = 1e-6
class SquashedGaussian(nn.Module):
    def __init__(self, dim, low, high, init_std = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.last_log_prob = None
        Normal.set_default_validate_args = False
        assert np.all(np.less(low, high))
        low = torch.from_numpy(np.array(low))
        high = torch.from_numpy(np.array(high))
        self.low = nn.Parameter(low, requires_grad=False)
        self.high = nn.Parameter(high, requires_grad=False)
        
    
    @property
    def mean(self):
        return self._squash(self.distribution.mean)
    
    @property
    def stddev(self):
        return self.distribution.stddev
    
    def update(self,logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

    def sample(self):
        samples = self.distribution.sample()
        squashed_samples = self._squash(samples)
        return squashed_samples

    def get_actions_log_prob(self,actions):
        # actions are fron [low, high]
        unsquashed_actions = self._unsquash(actions)
        actions_log_prob = self.distribution.log_prob(unsquashed_actions)
        actions_log_prob = torch.clamp(actions_log_prob,-100,100)
        actions_log_prob = actions_log_prob.sum(dim=-1)

        unsquashed_values_tanhd = torch.tanh(unsquashed_actions)
        log_prob = actions_log_prob - torch.sum(
            torch.log(1 - unsquashed_values_tanhd**2 + SMALL_NUMBER), dim=-1
        )
        self.last_log_prob = log_prob
        return log_prob

    def entropy(self):
        # squashed entropy estimated 
        return torch.mean(-self.last_log_prob, dim=0) 
    
        
    def _squash(self, raw_values):
        # From -inf, inf 
        # Returned values are within [low, high] (including `low` and `high`).
        squashed = ((torch.tanh(raw_values) + 1.0) / 2.0) * (
            self.high - self.low
        ) + self.low
        return torch.clamp(squashed, self.low, self.high)

    def _unsquash(self, values):
        # From [low, high] (including `low` and `high`)
        # Returned values are within [-inf, inf]. 
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        # Stabilize input to atanh.
        save_normed_values = torch.clamp(
            normed_values, -1.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER
        )
        unsquashed = torch.atanh(save_normed_values)
        return unsquashed

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            return str(obj)

class VecEnv(ABC):
    dt:float
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_obs_history: int 
    num_history:int
    obs_history_length: int
    num_actions: int
    num_rewards: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor 
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor # current episode duration
    extras: dict
    device: torch.device
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass
    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass
    @abstractmethod
    def reset_idx(self, env_ids: Union[list, torch.Tensor]):
        pass
    @abstractmethod
    def reset_history(self, env_ids: Union[list, torch.Tensor]):
        pass

    @abstractmethod
    def get_observations(self):
        pass
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass
    @abstractmethod
    def get_amp_observations(self)-> Union[torch.Tensor, None]:
        pass 
def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        # elif isinstance(val, tuple):
        #     #! Test ? 
        #     element = class_to_dict(val[0])
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return 


def dump_info( src_dict, tmp_dict):
    for k,v in tmp_dict.items():
        if k not in src_dict.keys():
            src_dict[k] = []
        src_dict[k].append(v)
    return src_dict

_EPS = np.finfo(float).eps * 4.0
def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out


class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip(
            (input - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(
            self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(
            self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp(
            (input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(
                expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(
                torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.normalize = torch.nn.functional.normalize

    def forward(self, x):
        x = self.normalize(x, dim=-1)
        return x

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def split_and_pad_small_traj(tensor,dones):
    dones = dones.clone()
    dones[-1] = 1
     # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)



def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)