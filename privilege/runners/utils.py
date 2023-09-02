from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union
import json 
import numpy as np 

# minimal interface of the environment

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            return str(obj)

class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_obs_history: int 
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
    def get_observations(self):
        pass
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass