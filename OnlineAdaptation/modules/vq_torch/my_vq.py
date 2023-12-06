from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from einops import rearrange, repeat, reduce, pack, unpack

from typing import Callable

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

def batched_embedding_v2(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b -> h b 1 d', d = dim)
    repeated_embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    quantize = repeated_embeds.gather(2, indices)
    return rearrange(quantize, 'h b 1 d -> h b d')

def ema_inplace(old, new, decay):
    old.mul_(decay).add_(new * (1 - decay))

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def orthogonal_init(*shape):
    t = torch.empty(shape)
    nn.init.orthogonal_(t)
    return t

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]
def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)
def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    #! t.shape (n_book, n_codebook, dim)
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)

def orthogonal_loss_fn_with_mask(t, mask):
    #! t.shape (n_book, n_codebook, dim)
    #! mask.shape (n_book, n_codebook)
    # perform othogonal loss on masked codes where mask==1
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    cosine_sim = cosine_sim * mask.unsqueeze(-1) * mask.unsqueeze(-2)
    n_effective = mask.sum(dim = -1)[0]
    return (cosine_sim ** 2).sum() / (h * n_effective ** 2) - (1 / n_effective)
    



def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).sqrt()
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    # x.shape = (n_book, n_code)
    denom = x.sum(dim = dim, keepdim = True)
    return (x + eps) / (denom + n_categories * eps) # (n_book,n_code), n_code 的概率

def simple_sample(logits,mask,temperature=1,deterministic=True,dim = -1):
    # loigits.shape = (n_head, bz, n_codebook)
    # mask.shape = (n_head, n_codebook)
    # sample logits with argmax from the mask==1 
    dtype, size = logits.dtype, logits.shape[dim]
    if deterministic:
        sampling_logits = logits
    else:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    if mask is not None:
        mask = repeat(mask, 'h c -> h b c', b = logits.shape[1])
        sampling_logits[~mask] = -1e10
    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)
    return ind, one_hot

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim = dim)
        π1 = (one_hot + (logits / temperature).softmax(dim = dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim = 1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim = dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot
class CodeBook(nn.Module):
    def __init__(self,
                 dim, # 这个 dim 就是分化好的
                 num_codebooks,
                 codebook_size,
                    ema_update = True,
                    decay = 0.8,
                    eps = 1e-5,
                    threshold_ema_dead_code = 2
                 ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        init_fn = uniform_init # 可能可以尝试有 pretrain 的初始化 
        self.gumbel_sample = gumbel_sample
        embed = init_fn(num_codebooks,codebook_size, dim)
        self.sample_codebook_temp = 1.0
        self.ema_update = ema_update
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code 
        self.reset_cluster_size = threshold_ema_dead_code
        self.sample_fn = batched_sample_vectors

        #! 是不是统计一下每个 embed 的选取个数, 可以查看 dead code
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        self.embed = nn.Parameter(embed)

        #! 算法需求
        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)

        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))
    
    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')
            
            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)


    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)
    
    @autocast(enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        # x.shape = 'h b n d', head 数量前置, b 是 batch size, n 感觉是辅助的 ?? d 是 dim
        needs_codebook_dim = x.ndim < 4
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, 'h ... -> h 1 ...')
        flatten,ps = pack_one(x,'h * d') # 这步的操作类似 从 h,b,n,d -> h, b*n,d
        embed = self.embed
        dist = -cdist(flatten, embed) # 越大相似度越高 

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp, training = self.training)
        embed_ind = unpack_one(embed_ind, ps, 'h *')
        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        
        if self.training and self.ema_update:
            cluster_size = embed_onehot.sum(dim = 1) 
            ema_inplace(self.cluster_size, cluster_size, self.decay)
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot) # 选择出来哪些 emb 被选中了 
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)
        
        dist = unpack_one(dist, ps, 'h * d')
        if needs_codebook_dim:
            quantize = rearrange(quantize, 'h 1 n d -> h n d')
            dist = rearrange(dist, 'h 1 n d -> h n d')
            embed_ind = rearrange(embed_ind, 'h 1 n -> h n')
        return quantize, embed_ind, dist
        
class SimpleCodeBook(nn.Module):
    def __init__(self,
                 dim, # 这个 dim 就是分化好的
                 num_codebooks,
                 codebook_size,
                    ema_update = True,
                    decay = 0.8,
                    eps = 1e-5,
                 ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        init_fn = orthogonal_init # 可能可以尝试有 pretrain 的初始化 
        self.sample_method = simple_sample
        embed = init_fn(num_codebooks,codebook_size, dim)
        self.sample_codebook_temp = 1.0
        self.ema_update = ema_update
        self.decay = decay
        self.eps = eps
        self.sample_fn = batched_sample_vectors

        #! 是不是统计一下每个 embed 的选取个数, 可以查看 dead code
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('valid_codebook', torch.ones(num_codebooks, codebook_size,dtype=torch.bool))
        self.register_buffer('embed_avg', embed.clone())
        self.embed = nn.Parameter(embed)

        #! 算法需求
        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)

        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))
    
    def unmask_all(self):
        self.valid_codebook.data.copy_(torch.ones_like(self.valid_codebook,dtype=torch.bool))
    def mask_percentage(self,percentage):
        assert percentage >= 0 and percentage <= 1
        if percentage == 0:
            self.unmask_all()
            return
        valid_num = int(self.codebook_size * (1-percentage))
        self.valid_codebook.data.copy_(torch.ones_like(self.valid_codebook,dtype=torch.bool))
        self.valid_codebook.data[:,valid_num:] = 0
    def random_mask(self):
        # 对每个头的 codebook 随机 mask 一些
        mask = torch.randint(low=0,high = 2, size = (self.codebook_size,),dtype=torch.bool)
        self.valid_codebook.data[:,mask] = 0
    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)
    
    @autocast(enabled = False)
    def forward(
        self,
        x
    ):
        # x.shape = 'h b d', head 数量前置, b 是 batch size, d 是 dim
        x = x.float()
        
        if self.ema_update:
            embed = self.embed.detach()
        else:
            embed = self.embed

        dist = -cdist(x, embed) # 越大相似度越高 

        embed_ind, embed_onehot = self.sample_method(logits = dist,
                                                     mask = self.valid_codebook,
                                                     temperature=1.0,
                                                     deterministic=True,
                                                     dim = -1)
        if self.training:
            quantize = einsum('h b c, h c d -> h b d', embed_onehot, embed)
        else:
            quantize = batched_embedding_v2(embed_ind, embed)
            # batch, dim = embed_ind.shape[1], embed.shape[-1]
            # indices = repeat(embed_ind, 'h b -> h b 1 d', d = dim)
            # repeated_embeds = repeat(embed, 'h c d -> h b c d', b = batch)
            # quantize = repeated_embeds.gather(2, indices)
            # quantize = rearrange(quantize, 'h b 1 d -> h b d')

        
        if self.training and self.ema_update:
            #! onehot.shape = (n_book,bz,n_code)-> n_code 激活的数量
            cluster_size = embed_onehot.detach().sum(dim = 1) 
            ema_inplace(self.cluster_size, cluster_size, self.decay) #! 统计每个 code 的激活数量均值
            embed_sum = einsum('h b d, h b c -> h c d', x, embed_onehot).detach() # 选择出来哪些 emb 被选中了 
            ema_inplace(self.embed_avg, embed_sum, self.decay) #! 统计用来激活 code 的emb 原来长什么样

            #! 这个平滑挺有趣的, 因为有的 code 没有被采样到,但是不能直接 + 1, 所以先算概率(Laplace平滑), 再乘上总数, 相当于略微的平分
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)
            #! 这一步是缓慢更新 emb
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
        
        return quantize, embed_ind, dist

class VectorQuantize(nn.Module):
    def __init__(self,
                 input_dim, n_head, codebook_size,
                 commitment_weight = 1.,
                 orthogonal_reg_weight = 0.,
                 ema_update = True,
                 decay = 0.8,
                 eps = 1e-5,):
        super().__init__()
        self.input_dim = input_dim 
        self.n_head = n_head
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.ema_update = ema_update

        codebook_dim = input_dim // n_head # 这个是每个 head 的维度
        assert codebook_dim * n_head == input_dim, 'input_dim must be divisible by n_head'
        self._codebook = SimpleCodeBook(
            dim = codebook_dim,
            num_codebooks = n_head,
            codebook_size = codebook_size,
            ema_update = ema_update,
            decay = decay,
            eps= eps,

        )
    
    def forward(self,x):
        # x.shape = 'bz, d',
        # x = rearrange(x,'bz d -> bz h d')
        ein_rhs_eq = 'h b d'
        x = rearrange(x, f"b (h d) -> {ein_rhs_eq}", h = self.n_head)
        # quantize
        quantize, embed_ind, distances = self._codebook(x)
        

        if self.training:
            # straight through
            if self.ema_update: 
                commit_quantize = torch.detach(quantize)
            else:
                commit_quantize = quantize
            quantize = x + (quantize - x).detach()

        embed_ind = rearrange(embed_ind, 'h b -> b h', h = self.n_head)
        

        loss = torch.tensor([0.], device = x.device, requires_grad = self.training)
        loss_info = {
            'commit_loss': 0.0,
            'orthogonal_reg_loss': 0.0,
        }

        if self.training:
            #! commit loss
            commit_loss = F.mse_loss(commit_quantize, x)
            loss = loss + commit_loss * self.commitment_weight
            loss_info['commit_loss'] = commit_loss.item()
            #! othogonal loss 
            if self.orthogonal_reg_weight > 0:
                orthogonal_reg_loss = orthogonal_loss_fn_with_mask(self._codebook.embed, self._codebook.valid_codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight
                loss_info['orthogonal_reg_loss'] = orthogonal_reg_loss.item() 
            else:
                loss_info['orthogonal_reg_loss'] = 0.0

        quantize = rearrange(quantize, 'h b d -> b (h d)', h = self.n_head)
        
        return quantize, embed_ind, loss,loss_info

    def get_info(self,x):
        ein_rhs_eq = 'h b d'
        x = rearrange(input, f"b (h d) -> {ein_rhs_eq}", h = self.n_head)
        # quantize
        quantize, embed_ind, distances = self._codebook(x)
        embed_ind = rearrange(embed_ind, 'h b -> b h', h = self.n_head)
        quantize = rearrange(quantize, 'h b d -> b (h d)', h = self.n_head)
        distances = rearrange(distances, 'h b d -> b h d')
        return quantize, embed_ind, distances