import torch.nn as nn 
import torch.nn.functional as F
import torch 
import numpy as np
import io 
def init_zero(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight,val = 0.0)
        m.bias.data.fill_(0.0)


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
    

class MLP(nn.Module):
    """
    MLP network
    """
    def __init__(self, input_dim, output_dim, 
                 hidden_dims=[256, 256, 256], activation= 'elu', 
                 output_activation='identity') -> None:
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        activation = get_activation(activation)
        layers.append(activation)
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(get_activation(output_activation))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
        
    

class TorchSquashedGaussian():
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.

    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """

    def __init__(self,mean,log_std,
                 low: float = -1.0,
                 high: float = 1.0):
        """Parameterizes the distribution via `inputs`.

        Args:
            low (float): The lowest possible sampling value
                (excluding this value).
            high (float): The highest possible sampling value
                (excluding this value).
        """
        # Split inputs into mean and log(std).
        # Clip `scale` values (coming from NN) to reasonable values.
        super().__init__()  
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        self.dist = torch.distributions.normal.Normal(mean, std)
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high
    def deterministic_sample(self):
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample
    def sample(self):
        # Use the reparameterization version of `dist.sample` to allow for
        # the results to be backprop'able e.g. in a loss term.
        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)
        return self.last_sample
    def logp(self, x) :
        # Unsquash values (from [low,high] to ]-inf,inf[)
        unsquashed_values = self._unsquash(x)
        # Get log prob of unsquashed values from our Normal.
        log_prob_gaussian = self.dist.log_prob(unsquashed_values)
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1)
        # Get log-prob for squashed Gaussian.
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - torch.sum(
            torch.log(1 - unsquashed_values_tanhd**2 + 1e-6), dim=-1)
        return log_prob

    def _squash(self, raw_values):
        # Returned values are within [low, high] (including `low` and `high`).
        squashed = ((torch.tanh(raw_values) + 1.0) / 2.0) * \
            (self.high - self.low) + self.low
        return torch.clamp(squashed, self.low, self.high)

    def _unsquash(self, values) :
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - \
                        1.0
        # Stabilize input to atanh.
        save_normed_values = torch.clamp(normed_values, -1.0 + 1e-6,
                                         1.0 - 1e-6)
        unsquashed = torch.atan(save_normed_values)
        return unsquashed
    
    @property
    def mean(self):
        return self._squash(self.dist.mean)
    
def load_script_model(path,device="cpu"):
    with open(path, 'rb') as m:
        models = io.BytesIO(m.read())
    net = torch.jit.load(models, map_location=torch.device(device))
    return net

def save_script_model(path,model):
    script_net = torch.jit.script(model)
    torch.jit.save(script_net, path)