import numpy as np
import torch
import os 
from typing import Callable, List

import copy 
from sklearn.cluster import KMeans
from torch.distributions.dirichlet import Dirichlet

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower




