from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from einops import rearrange, repeat, reduce, pack, unpack

from typing import Callable