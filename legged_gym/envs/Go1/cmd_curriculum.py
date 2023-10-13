import numpy as np
from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym.envs.Go1.basic_config import BasicCfg  
import torch
from matplotlib import pyplot as plt


def is_met(scale, l2_err, threshold):
    return (l2_err / scale) < threshold


def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    # metric_cache[target_key][env_id] / ep_len
    scale = 1
    l2_err = 0
    return is_met(scale, l2_err, threshold)


class Curriculum:
    def set_to(self, low, high, value=1.0):
        inds = np.logical_and(
            self.grid >= low[:, None],
            self.grid <= high[:, None]
        ).all(axis=0)

        assert len(inds) != 0, "You are intializing your distribution with an empty domain!"

        self.weights[inds] = value

    def __init__(self, seed, **key_ranges):
        self.rng = np.random.RandomState(seed)

        self.cfg = cfg = {}
        self.indices = indices = {}
        for key, v_range in key_ranges.items():
            bin_size = (v_range[1] - v_range[0]) / v_range[2]
            #! command 的离散化
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2])
            #! bin size is not the same as the range
            indices[key] = np.linspace(0, v_range[2]-1, v_range[2])

        self.lows = np.array([range[0] for range in key_ranges.values()])
        self.highs = np.array([range[1] for range in key_ranges.values()])

        # self.bin_sizes = {key: arr[1] - arr[0] for key, arr in cfg.items()}
        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}

        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self._idx_grid = np.stack(np.meshgrid(*indices.values(), indexing='ij'))
        self.keys = [*key_ranges.keys()]
        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])
        # self.grid = np.stack([params.flatten() for params in raw_grid])

        self._l = l = len(self.grid[0])
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}

        self.weights = np.zeros(l) #! l 是所有可能的组合的个数
        self.indices = np.arange(l)#! 只是为了 加权采样

    def __len__(self):
        return self._l

    def __getitem__(self, *keys):
        pass

    def update(self, **kwargs):
        # bump the envelop if
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """default to uniform"""
        if low is not None and high is not None: # if bounds given
            valid_inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else: # if no bounds given
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())

        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = np.array([*self.bin_sizes.values()])
        #! 假如 选定 [1,2], binsize 是 [0.5,1.0], 那么就是在 low = [0.75, 1.5] 到 high = [1.25, 2.5] 之间随机
        low, high = centroids + bin_sizes / 2, centroids - bin_sizes / 2
        return self.rng.uniform(low, high)#.clip(self.lows, self.highs)

    def sample(self, batch_size, low=None, high=None):
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


class SumCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.success = np.zeros(len(self))
        self.trials = np.zeros(len(self))

    def update(self, bin_inds, l1_error, threshold):
        is_success = l1_error < threshold
        self.success[bin_inds[is_success]] += 1
        self.trials[bin_inds] += 1

    def success_rates(self, *keys):
        s_rate = self.success / (self.trials + 1e-6)
        s_rate = s_rate.reshape(list(self.ls.values()))
        marginals = tuple(i for i, key in enumerate(self.keys) if key not in keys)
        if marginals:
            return s_rate.mean(axis=marginals)
        return s_rate


class RewardThresholdCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.episode_reward_lin = np.zeros(len(self))
        self.episode_reward_ang = np.zeros(len(self))
        self.episode_lin_vel_raw = np.zeros(len(self))
        self.episode_ang_vel_raw = np.zeros(len(self))
        self.episode_duration = np.zeros(len(self))

    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)

        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)
        ).all(axis=0)

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):

        is_success = 1.
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
        else:
            is_success = np.array(is_success.bool())

        # if len(is_success) > 0 and is_success.any():
        #     print("successes")
        #! 这个课程学习的机制很 naive 的, 给 success 的增加 weights, 然后给周围的增加 weights, 
        #! 可能探索机制是在于, 会逐渐 sample success 周围的
        #! 衡量每一组 commands 是否学好了, 就看 weights 是不是都是 1 了
        #! 强调一下: 这个 sample 机制可能会反复采样已经训练过的环境
        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        for adjacent in adjacents:
            #print(adjacent)
            #print(self.grid[:, adjacent])
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()


class RewardThresholdCurriculum_v2(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.episode_reward_lin = np.zeros(len(self))
        self.episode_reward_ang = np.zeros(len(self))
        self.episode_lin_vel_raw = np.zeros(len(self))
        self.episode_ang_vel_raw = np.zeros(len(self))
        self.episode_duration = np.zeros(len(self))

        self.success_counts = np.zeros_like(self.weights)
        self.sampled_counts = np.zeros_like(self.weights)

    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)

        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)
        ).all(axis=0)

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):

        is_success = 1.
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
            is_failure = np.array([True] * len(bin_inds))
        else:
            tmp_success = is_success
            is_success = np.array(tmp_success.bool())
            is_failure = np.array(~tmp_success.bool())

        # if len(is_success) > 0 and is_success.any():
        #     print("successes")
        #! 这个课程学习的机制很 naive 的, 给 success 的增加 weights, 然后给周围的增加 weights, 
        #! 可能探索机制是在于, 会逐渐 sample success 周围的
        #! 衡量每一组 commands 是否学好了, 就看 weights 是不是都是 1 了
        #! 强调一下: 这个 sample 机制可能会反复采样已经训练过的环境
        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        self.weights[bin_inds[is_failure]] = np.clip(self.weights[bin_inds[is_failure]] - 0.1, 0.1, 1)
        
        self.success_counts[bin_inds[is_success]] += 1
        self.sampled_counts[bin_inds] += 1


        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        for adjacent in adjacents:
            #print(adjacent)
            #print(self.grid[:, adjacent])
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()
    
    def sample_bins(self, batch_size, low=None, high=None):
        """default to uniform"""
        if low is not None and high is not None: # if bounds given
            valid_inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else: # if no bounds given
            cur_weights = self.weights  
            success_rate = self.success_counts / (self.sampled_counts + 1e-6)
            wraped_weight = cur_weights / (np.sqrt(success_rate) + 1e-6)
            weight = wraped_weight / wraped_weight.sum()
            inds = self.rng.choice(self.indices, batch_size, p= weight)

        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = np.array([*self.bin_sizes.values()])
        #! 假如 选定 [1,2], binsize 是 [0.5,1.0], 那么就是在 low = [0.75, 1.5] 到 high = [1.25, 2.5] 之间随机
        low, high = centroids + bin_sizes / 2, centroids - bin_sizes / 2
        return self.rng.uniform(low, high)#.clip(self.lows, self.highs)

    def sample(self, batch_size, low=None, high=None):
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


class GridAdaptiveCurriculum:
    """
    1. 确定每个 cmd 的上下限, 以及 bin size, 最后通过遍历 bin 来实现策略评估
    """
    def __init__(self,cfg:BasicCfg,seed = 1):
        self.rng = np.random.RandomState(seed)
        cmd_curriculum_cfg = cfg.commands.cmd_cfg
        idx = list(cmd_curriculum_cfg.keys())
        idx.sort()
        self.lows = np.array([cmd_curriculum_cfg[i]['limit_low'] for i in idx])
        self.highs = np.array([cmd_curriculum_cfg[i]['limit_high'] for i in idx])
        
        self.bin_sizes = {}
        self.bin_cmd_values = {}
        self.bin_cmd_indices = {}
        self.local_ranges = []
        init_low,init_high = [],[]
        for i in idx:
            i = int(i)
            cmd = cmd_curriculum_cfg[i] 
            self.bin_sizes[i] = (cmd['limit_high'] - cmd['limit_low']) / cmd['num_bins']
            self.bin_cmd_values[i] = np.linspace(cmd['limit_low'] + self.bin_sizes[i] / 2, cmd['limit_high'] - self.bin_sizes[i] / 2, cmd['num_bins'])
            self.bin_cmd_indices[i] = np.linspace(0, cmd['num_bins']-1, cmd['num_bins'])
            self.local_ranges.append(cmd['local_range'])
            init_low.append(cmd['init_low'])
            init_high.append(cmd['init_high'])
        init_high,init_low = np.array(init_high),np.array(init_low)
        self.local_ranges = np.array(self.local_ranges) 
        self._raw_bin_values_grid = np.stack(np.meshgrid(*self.bin_cmd_values.values(), indexing='ij'))
        self._raw_bin_idx_grid = np.stack(np.meshgrid(*self.bin_cmd_indices.values(), indexing='ij'))
        self._np_bin_size = np.array([*self.bin_sizes.values()])
        self.keys = [*cmd_curriculum_cfg.keys()]
        n_cmd = len(self.keys)
        self.bin_values_grid = self._raw_bin_values_grid.reshape([n_cmd, -1]).T
        self.bin_idx_grid = self._raw_bin_idx_grid.reshape([n_cmd, -1]).T

        self.n_combinations = len(self.bin_values_grid)
        self.weights = np.zeros(self.n_combinations)
        self.indices = np.arange(self.n_combinations)
        inds = np.logical_and(
            self.bin_values_grid >= init_low[None, :],
            self.bin_values_grid <= init_high[None, :]
        ).all(axis=1)
        assert len(inds) > 0, "You are intializing your distribution with an empty domain!"
        self.weights[inds] = 1.0 

        self.success_num = np.zeros(self.n_combinations)
        self.totoal_num = np.zeros(self.n_combinations)


    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.ones(self.bin_values_grid.shape[1]) * ranges
        adjacent_inds = np.logical_and(
            self.bin_values_grid[:,None,:].repeat(bin_inds.shape[0],axis=1) >= self.bin_values_grid[None,bin_inds,:] - ranges,
            self.bin_values_grid[:,None,:].repeat(bin_inds.shape[0],axis=1) <= self.bin_values_grid[None,bin_inds,:] + ranges,
            ).all(axis=-1)
        list_adjacent_inds = [adjacent_inds[:,i].nonzero()[0] for i in range(adjacent_inds.shape[1])]
        return list_adjacent_inds
    def sample_bins(self,batch):
        """
        1. 只是 sample 代采样的 bin, 不是具体数值
        """
        inds = self.rng.choice(self.indices, batch, p=self.weights / self.weights.sum())
        return self.bin_values_grid[inds], inds
    def sample_uniform_from_bin(self,center):
        # center.shape ( n_batch, n_cmd)
        low,high = center-self._np_bin_size/2,center+self._np_bin_size/2
        return self.rng.uniform(low,high,size = center.shape)
    def sample(self,batch_size):
        center,inds = self.sample_bins(batch_size)
        cmd = self.sample_uniform_from_bin(center)
        return cmd,inds 
    
    def _update(self,bin_inds, task_rewards, success_thresholds):
        # print("CMD Debug: We resample !", task_rewards, success_thresholds)
        is_success = 1.
        for task_reward, success_threshold in zip(task_rewards, success_thresholds): # 需要多个 threshold 来判断是否成功
            is_success = is_success * (task_reward > success_threshold).cpu()

        is_success = is_success.reshape(-1).numpy()
        is_success = is_success.nonzero()[0]
        # print(bin_inds.shape, is_success.shape)
        # print(bin_inds,is_success)
        # print("CMD Debug: We resample !", bin_inds[is_success])
        self.success_num[bin_inds[is_success]] += 1.0
        self.totoal_num[bin_inds] += 1.0
    
    def update(self):
        
        success_rate = self.success_num / (self.totoal_num + 1e-6)
        is_success = 1.
        is_success = is_success * (success_rate > 0.5)
        is_success = is_success.nonzero()[0]

        self.weights[is_success] = np.clip(self.weights[is_success] + 0.2, 0, 1)
        adjacents = self.get_local_bins(is_success, ranges=self.local_ranges)

        for adjacent in adjacents:
            self.weights[adjacent] = np.clip(self.weights[adjacent] + 0.2, 0, 1)

        self.success_num = np.zeros(self.n_combinations)
        self.totoal_num = np.zeros(self.n_combinations)
        # print("CMD Debug: We Update !", is_success)
    # def update(self,bin_inds, task_rewards, success_thresholds):
    #     print("Debug Cmd Curriculum: ", task_rewards, success_thresholds)
    #     is_success = 1.
    #     for task_reward, success_threshold in zip(task_rewards, success_thresholds): # 需要多个 threshold 来判断是否成功
    #         is_success = is_success * (task_reward > success_threshold).cpu() 
    #     if len(success_thresholds) == 0:
    #         is_success = np.array([False] * len(bin_inds))
    #     else:
    #         is_success = np.array(is_success.bool())
    #     print("Debug Cmd Curriculum 2: ", is_success)
    #     self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
    #     list_adjacents = self.get_local_bins(bin_inds[is_success], ranges=self.local_ranges)
    #     for adjacent in list_adjacents:
    #         self.weights[adjacent] = np.clip(self.weights[adjacent] + 0.2, 0, 1)    



if __name__ == '__main__':
    r = RewardThresholdCurriculum(100, x=(-1, 1, 5), y=(-1, 1, 2), z=(-1, 1, 11))

    assert r._raw_grid.shape == (3, 5, 2, 11), "grid shape is wrong: {}".format(r.grid.shape)

    low, high = np.array([-1.0, -0.6, -1.0]), np.array([1.0, 0.6, 1.0])

    # r.set_to(low, high, value=1.0)

    adjacents = r.get_local_bins(np.array([10, ]), range=0.5)
    for adjacent in adjacents:
        adjacent_inds = np.array(adjacent.nonzero()[0])
        print(adjacent_inds)
        r.update(bin_inds=adjacent_inds, lin_vel_rewards=np.ones_like(adjacent_inds),
                 ang_vel_rewards=np.ones_like(adjacent_inds), lin_vel_threshold=0.0, ang_vel_threshold=0.0,
                 local_range=0.5)

    samples, bins = r.sample(10_000)

    plt.scatter(*samples.T[:2])
    plt.show()
