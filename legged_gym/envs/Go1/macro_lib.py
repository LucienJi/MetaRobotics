from legged_gym.utils.helpers import class_to_dict
import numpy as np
import torch 

def _sliding_window(contacts_force):
        # contacts.shape = (n_steps, n_env , n_foot)
        contacts = contacts_force > 1.0 # 1.0 is a threshold 
        contacts_filt = np.zeros_like(contacts) 
        for i in range(contacts.shape[0]):
            if i == 0:
                contacts_filt[i] = contacts[i]
            else:
                contacts_filt[i] = np.logical_or(contacts[i],contacts[i-1]) 
        return contacts_filt


class RewardScaleManager:
    def __init__(self, cfg, device) -> None:
        self.reward_scales = class_to_dict(cfg.reward_scales) # from env 
        self.reward_ranges = class_to_dict(cfg.reward_ranges) # from env
        self.device = device
        self.cfg = cfg 
        self.parse_scaled_ranges()
    
    
    def parse_scaled_ranges(self):
        #!不重复计算 reward function 这个还是交给 env
        ranges = list(self.reward_ranges.keys())
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0 and (key not in ranges):
                self.reward_scales.pop(key)
        reward_names = list(set(self.reward_scales.keys()).union(set(self.reward_ranges.keys())))
        reward_name, task_name = [],[] 
        for name in reward_names:
            if name in ['freq','phases','foot_height']:
                task_name.append(name)
            else:
                reward_name.append(name)
        self.reward_names = reward_name 
        self.task_names = task_name
        self.total_names = self.reward_names + self.task_names

        self.fixed_reward_names = []
        self.fixed_reward_scales = {}
        self.unfixed_reward_names = []
        self.unfixed_task_names = task_name  
        self.unfixed_total_names = []
        self.reward_factor = {}
        lbs,ubs = [],[]
        self._preweight2weight = {}
        self.preweightSampler = {}
        for name in self.total_names:
            if name == 'termination':
                continue
            if name in self.reward_ranges.keys():
                if name not in ['freq','phases','foot_height']:
                    self.unfixed_reward_names.append(name)
                self.unfixed_total_names.append(name)
                v,f = self.reward_ranges[name][0],self.reward_ranges[name][1]
                self.reward_factor[name] = {}
                self.reward_factor[name]['lower_bound'] = v[0]
                self.reward_factor[name]['upper_bound'] = v[-1]
                self.reward_factor[name]['np_candidate'] = v
                self.reward_factor[name]['scale_candidate'] = torch.from_numpy((v - v[0])/ (v[-1] - v[0])).to(self.device)
                self._preweight2weight[name] = f
                lbs.append(v[0])
                ubs.append(v[-1])
                self.preweightSampler[name] = lambda n,name:np.random.choice(a = self.reward_factor[name]['np_candidate'], size = (n,),replace=True) 
            else:
                self.fixed_reward_names.append(name)
                self.fixed_reward_scales[name] = self.reward_scales[name]

        self.num_fixed_rewards = len(self.fixed_reward_names)
        self.num_unfixed_rewards = len(self.unfixed_reward_names) #! unfixed reward 也包含 unfixed task 
        self.num_task = len(self.task_names)
        self.num_unfixed_total = len(self.unfixed_total_names)

        self.reward_lbs = torch.tensor(lbs).to(self.device).reshape(1,-1)
        self.reward_ubs = torch.tensor(ubs).to(self.device).reshape(1,-1)

    def preweight2weight(self,preweight):
        if isinstance(preweight,np.ndarray):
            preweight = torch.from_numpy(preweight).to(self.device)
        weight = torch.zeros_like(preweight)
        assert preweight.shape[1] == self.num_unfixed_total, print("Shape Inconsistent")
        for i in range(self.num_unfixed_total):
            name = self.unfixed_total_names[i]
            weight[:,i] = self._preweight2weight[name](preweight[:,i])
        return weight


    def preweight2scale(self, preweight):
        if not isinstance(preweight,torch.Tensor):
            preweight = torch.from_numpy(preweight).to(self.device)
        # print("Shape Check: ", preweight.shape, self.reward_lbs.shape, self.reward_ubs.shape)
        scales = (preweight - self.reward_lbs)/(self.reward_ubs - self.reward_lbs + 1e-8)
        return scales 
    
    def scale2preweight(self,scales):
        if isinstance(scales,np.ndarray):
            scales = torch.from_numpy(scales).to(self.device) 
        preweight = scales * (self.reward_ubs - self.reward_lbs + 1e-8) + self.reward_lbs 
        return preweight 
    
    def sample_weights(self, num_envs):
        '''
        1. 根据个数需求生成随机 weight 
        '''
        weights = []
        scales = []
        preweights = []
        for i in range(self.num_unfixed_total):
            name = self.unfixed_total_names[i]
            preweight = self.preweightSampler[name](num_envs,name)
            preweights.append(preweight)
            weight = self._preweight2weight[name](preweight)
            weights.append(weight)
        preweights = np.stack(preweights,axis = -1 )
        weights = np.stack(weights,axis = -1)
        weights = torch.from_numpy(weights).to(self.device)
        scales = self.preweight2scale(preweights)
        return scales, weights
    
    def scale2weight(self,scales, discrete = False):
        if isinstance(scales,np.ndarray):
            scales = torch.from_numpy(scales).to(self.device)
        if discrete :
            # print("RAW Scale: ", scales)
            scales = self.discretize(scales)
            # print("After: ", scales)
        preweight = self.scale2preweight(scales) 
        return scales, self.preweight2weight(preweight)
    
    def discretize(self,scales):
        new_scales = []
        if isinstance(scales,np.ndarray):
            scales = torch.from_numpy(scales).to(self.device)
        # scales.shape = (n_env, n_rew) 
        n_env = scales.shape[0]
        for i in range(self.num_unfixed_total):
            name = self.unfixed_total_names[i] 
            candidate = self.reward_factor[name]['scale_candidate'].unsqueeze(0)
            scale = scales[:,i].unsqueeze(-1)
            diff = torch.abs(scale - candidate) # (n_env, n_candidate) 
            idx = torch.argmin(diff, dim = -1)
            candidate = torch.gather(candidate, dim = -1, index = idx.unsqueeze(0)).squeeze()
            new_scales.append(candidate)
        new_scales = torch.stack(new_scales,dim = -1)
        
        return new_scales



class MacroScoreManager:
    def __init__(self,env,macro_score_list):
        self.env = env
        self.macro_score_list = macro_score_list
        self.device = env.device 
        self.num_envs = env.num_envs
        self.scores = dict()
        self.counts = dict()

        self.macro_score_names = []
        self.macro_score_functions = []
        self.stride = None
        for macro_score_name in self.macro_score_list:
            self.macro_score_names.append(macro_score_name)
            self.macro_score_functions.append(getattr(self,"_compute_" + macro_score_name))

            if macro_score_name == 'stride':
                self.stride = StrideStats(self.num_envs,env.feet_indices,env.foot_positions,env.device,min_stride_count=10)
            self.scores[macro_score_name] = torch.zeros(self.num_envs,device = self.device,dtype = torch.float,requires_grad=False)
            self.counts[macro_score_name] =  torch.zeros(self.num_envs,device = self.device,dtype = torch.float,requires_grad=False)
        
        self.num_styles = len(self.macro_score_list)
        
    def reset(self,env_ids):
        for macro_score_name in self.macro_score_list:
            self.scores[macro_score_name][env_ids] = 0.
            self.counts[macro_score_name][env_ids] = 0.
        if self.stride is not None:
            self.stride.reset(env_ids,self.env.foot_positions)

    def update_macro_score(self):
        for i in range(self.num_styles):
            name = self.macro_score_names[i]
            if name != 'stride':
                score = self.macro_score_functions[i]()
                self._update_macro_score(name,score)
            else:
                score = self.macro_score_functions[i]() 
                # print("Stride Check: ", score)
                if score is not None:
                    self._update_macro_score(name,score)
    
    def get_macro_score(self):
        info = {}
        for name in self.macro_score_list:
            info[name] = self.scores[name] / (self.counts[name] + 1e-8)
        # self.reset(torch.arange(self.num_envs, device=self.device))
        return info
    
    # ------------ 以下是 Macro Score function 的计算 -------------- #
    def _compute_cot(self):
        m = 13.1
        g = 9.8
        power = torch.sum(torch.clamp(self.env.dof_vel * self.env.torques, min = 0.0),dim = 1)
        # power = torch.sum(self.torques * self.dof_vel,dim = 1)
        vel_norm = torch.norm(self.env.base_lin_vel[:,0:2],p=2,dim = 1)
        cot = power / (m * g * vel_norm) #! 越小越好
        return cot 
    def _compute_tracking(self):
        lin_vel_error = torch.sum(torch.square(self.env.commands[:,:2] - self.env.base_lin_vel[:,0:2]),dim = 1)
        ang_vel_error = torch.square(self.env.commands[:,2] - self.env.base_ang_vel[:,2])
        return lin_vel_error + ang_vel_error 
        
    def _compute_stride(self):
        self.__update_stride_stats()
        self.stride.update()
        stride_duration_cv, stride_length_cv = self.stride.get_variability() # 数值越大， 越不稳定 
        if stride_duration_cv is None:
            return None 
        return stride_duration_cv + stride_length_cv 
    
    def __update_stride_stats(self):
        contact = self.env.contact_forces[:,self.env.feet_indices,2] > 1.0 
        contact = torch.logical_or(contact,self.env.last_contacts) ## filtered contact 
        
        # 判断 feet 是否是第一次 contact， 如果是，需要把对应 foot 的 air 加入到 stride 中
        # 同时， 假如 contact = True， 就可以把 feet time 置零， 不然就默认加上 dt
        first_contact = (self.stride.feet_time_in_air > 0.0) * contact ## 满足两个条件， 1. feet time in air > 0.0, 2. contact = True
        # first_contact.shape = (num_envs,num_feet)
        self.stride.feet_time_in_air += self.env.dt 
        self.stride.tmp_stride_duration += self.stride.feet_time_in_air * first_contact # 只有 True 的地方才会加上 dt
        self.stride.tmp_stride_count += torch.ones_like(self.stride.tmp_stride_count) * first_contact
        
        self.stride.feet_time_in_air *= ~contact # 如果 contact = True, 就把 feet time 置零

        delta_feet_pos = torch.norm(self.env.foot_positions - self.stride.last_foot_pos , dim = -1)
        self.stride.tmp_stride_length += delta_feet_pos * first_contact # 只有 True 的地方才会加上步长
        
        new_last_foot_post = first_contact.unsqueeze(-1) * self.env.foot_positions + ~first_contact.unsqueeze(-1) * self.stride.last_foot_pos # 如果是第一次 contact, 就更新 last_foot_pos
        self.stride.last_foot_pos[:] = new_last_foot_post[:]
        # print("Stride Time: ",self.stride.tmp_stride_count)
        
    def _update_macro_score(self,name, macro_score):
        self.scores[name] += macro_score
        self.counts[name] += 1

    @staticmethod
    def compute_tracking(data):
        """
        input: 
            1. data : dictionary
            2. shape = (n_steps, n_envs, dims)
        """
        cmd_x,cmd_y = data["command_x"],data["command_y"]
        cmd_yaw = data["command_yaw"]

        base_vel_x,base_vel_y,base_vel_yaw = data["base_vel_x"],data["base_vel_y"],data["base_vel_yaw"]
        lin_vel_error = np.square(cmd_x - base_vel_x) + np.square(cmd_y - base_vel_y)
        ang_vel_error = np.square(cmd_yaw - base_vel_yaw)  
        lin_vel_error = np.mean(lin_vel_error,axis = 0) 
        ang_vel_error = np.mean(ang_vel_error,axis = 0)
        return lin_vel_error + ang_vel_error
    @staticmethod
    def compute_cot(data):
        m,g = 13.1, 9.8 
        dof_vel = data['dof_vel'] # shape = (n_steps, n_envs, n_dofs)
        torques = data['dof_torque'] # shape = (n_steps, n_envs, n_dofs)
        vel = np.stack([data['base_vel_x'], data['base_vel_y']],axis = -1) 
        vel_norm = np.linalg.norm(vel,axis = -1,ord=2) # shape = (n_steps, n_envs)
        power = np.sum(np.clip(dof_vel * torques , a_min = 0.0, a_max= None), axis = -1) # shape = (n_steps, n_envs)
        cot = power / (m*g*vel_norm) # shape = (n_steps, n_envs)
        cot = np.mean(cot,axis = 0) # shape = (n_envs) 
        return cot  
    # @staticmethod 
    # def compute_stride(data,dt = 0.02):
    #     contacts = _sliding_window(data['contact_force_z'])
    #     dones = data['dones']
    #     foot_pos = data['foot_position']
    #     n_env = contacts.shape[1] 

    #     stride_duration_mean = np.zeros(shape=(n_env,4))
    #     stride_duration_std = np.zeros(shape=(n_env,4))
    #     stride_length_mean = np.zeros(shape=(n_env,4))
    #     stride_length_std = np.zeros(shape=(n_env,4))

    #     for foot_id in range(4):
    #         for env_id in range(n_env):
    #             feet_in_air = []
    #             feet_stride_length = []
    #             last_first_contact_idx = -1
    #             first_contact_idx = -1
    #             for i in range(contacts.shape[0]):
    #                 if i == 0:
    #                     first_contact = contacts[i,env_id,foot_id] 
    #                 else: 
    #                     first_contact = contacts[i,env_id,foot_id] * (1 - contacts[i-1,env_id,foot_id]) 
    #                 if first_contact:
    #                     last_first_contact_idx = first_contact_idx
    #                     first_contact_idx = i
                    
    #                 if dones[i,env_id]:
    #                     last_first_contact_idx = -1
    #                 if first_contact and last_first_contact_idx != -1:
    #                     delta_step = first_contact_idx - last_first_contact_idx
    #                     feet_in_air.append(delta_step * dt)
    #                     feet_stride_length.append(np.linalg.norm((foot_pos[first_contact_idx,env_id,foot_id] - foot_pos[last_first_contact_idx,env_id,foot_id]),ord=2))
    #             feet_in_air_mean, feet_in_air_std = np.mean(feet_in_air), np.std(feet_in_air)
    #             feet_stride_length_mean, feet_stride_length_std = np.mean(feet_stride_length), np.std(feet_stride_length)
    #             stride_duration_mean[env_id,foot_id] = feet_in_air_mean
    #             stride_duration_std[env_id,foot_id] = feet_in_air_std
    #             stride_length_mean[env_id,foot_id] = feet_stride_length_mean
    #             stride_length_std[env_id,foot_id] = feet_stride_length_std
    #     stride_duration_cv = stride_duration_std / (stride_duration_mean + 1e-6)
    #     stride_length_cv = stride_length_std / (stride_length_mean + 1e-6)
    #     #! 计算腿的 CV 综合
    #     stride_duration_cv = np.mean(stride_duration_cv,axis=-1)
    #     stride_length_cv = np.mean(stride_length_cv,axis=-1)
    #     return stride_duration_cv + stride_length_cv

    @staticmethod
    def compute_stride(data,dt = 0.02):
        contacts = _sliding_window(data['contact_force_z'])
        dones = data['dones']
        foot_pos = data['foot_position']
        foot_number = foot_pos.shape[2]
        n_envs = contacts.shape[1]
        n_steps = contacts.shape[0] 
        first_contacts = np.zeros_like(contacts)
        first_contacts[:,0,:] = contacts[:,0,:]
        first_contacts[:,1:,:] = contacts[:,1:,:] * ( 1 - contacts[:,:-1,:] )

        stride_duration_mean = np.zeros(shape=(n_envs,))
        stride_duration_std = np.zeros(shape=(n_envs,))

        stride_length_mean = np.zeros(shape=(n_envs,))
        stride_length_std = np.zeros(shape=(n_envs,))

        for i in range(n_envs):
            delta_pos_norm = []
            stride_duration = [] 
            for j in range(foot_number):
                tmp_first_contact = first_contacts[:,i,j]
                tmp_foot_pos = foot_pos[:,i,j,:]
                tmp_first_contact_foot_pos = tmp_foot_pos[tmp_first_contact]
                delta_pos = tmp_first_contact_foot_pos[1:] - tmp_first_contact_foot_pos[:-1]
                delta_pos_norm_ = np.linalg.norm(delta_pos, axis=1, ord=2)
                delta_pos_norm_ = delta_pos_norm_[delta_pos_norm_ < 1.0]
                delta_pos_norm.extend(delta_pos_norm_)
                tmp_first_contact_ = np.where(tmp_first_contact)[0]
                stride_duration.extend((tmp_first_contact_[1:] - tmp_first_contact_[:-1]) * dt)
            delta_pos_norm = np.array(delta_pos_norm)   
            stride_duration = np.array(stride_duration)
            stride_duration_mean[i] = np.mean(stride_duration)
            stride_duration_std[i] = np.std(stride_duration)
            stride_length_mean[i] = np.mean(delta_pos_norm)
            stride_length_std[i] = np.std(delta_pos_norm)
        stride_duration_cv = stride_duration_std / (stride_duration_mean + 1e-6)
        stride_length_cv = stride_length_std / (stride_length_mean + 1e-6)
        stride_duration_cv = np.nan_to_num(stride_duration_cv,nan = 0.0)
        stride_length_cv = np.nan_to_num(stride_length_cv,nan = 0.0)
        return stride_duration_cv + stride_length_cv




    @staticmethod
    def eval_macro_score(data):
        # data.shape = (n_step,n_env, dims)
        macro_score = {}
        macro_score['tracking'] = MacroScoreManager.compute_tracking(data)
        macro_score['cot'] = MacroScoreManager.compute_cot(data)
        macro_score['stride'] = MacroScoreManager.compute_stride(data)
        return macro_score


class StrideStats:
    def __init__(self,num_envs,feet_indices, initial_feet_pos,device,min_stride_count = 10) -> None:
        self.num_envs = num_envs
        self.list_stride_duration = [] # add duration with shape (num_envs,)， ndarray
        self.list_stride_length = [] # add length with shape (num_envs,)， ndarray
        self.tmp_stride_duration = torch.zeros(num_envs,feet_indices.shape[0],device = device,dtype = torch.float, requires_grad=False)
        self.tmp_stride_length = torch.zeros(num_envs,feet_indices.shape[0],device = device,dtype = torch.float, requires_grad=False)
        self.tmp_stride_count = torch.zeros(num_envs,feet_indices.shape[0],device = device,dtype = torch.float, requires_grad=False)   

        # feet time in air 
        self.feet_time_in_air = torch.zeros(num_envs,feet_indices.shape[0],device = device,dtype = torch.float, requires_grad=False)

        self.feet_indices = feet_indices

        self.last_foot_pos = torch.zeros(num_envs,feet_indices.shape[0],3,device = device,dtype = torch.float, requires_grad=False)
        self.last_foot_pos[:] = initial_feet_pos[:]
        self.min_stride_count = min_stride_count
        self.device = device

    def reset(self,env_ids,initial_feet_pos):
        self.tmp_stride_duration[env_ids] = 0.0
        self.tmp_stride_length[env_ids] = 0.0
        self.tmp_stride_count[env_ids] = 0.0
        self.feet_time_in_air[env_ids] = 0.0
        self.last_foot_pos[env_ids] = initial_feet_pos[env_ids]
        
        
    
    def update(self):
        '''
        我们希望频繁 call update ;
        更少的 call get variability
        '''
        count = torch.sum(self.tmp_stride_count,dim = 1) # shape = (num_envs,)
        if torch.all(count > 0.0): # 起码每个环境都有 stride 了
            
            self.list_stride_duration.append((torch.sum(self.tmp_stride_duration,dim = 1) / count).cpu().numpy())
            self.list_stride_length.append((torch.sum(self.tmp_stride_length,dim = 1) / count).cpu().numpy())
            self.tmp_stride_count[:] = 0.0
            self.tmp_stride_duration[:] = 0.0
            self.tmp_stride_length[:] = 0.0
            # print("ADD: ", len(self.list_stride_duration))
    def get_variability(self):
        if len(self.list_stride_duration) > self.min_stride_count:
            duration = np.stack(self.list_stride_duration,axis = 1)
            length = np.stack(self.list_stride_length,axis = 1)
            duration_cv = np.std(duration,axis = 1) / np.mean(duration,axis = 1)
            length_cv = np.std(length,axis = 1) / np.mean(length,axis = 1)
            self.list_stride_duration = []
            self.list_stride_length = []
            duration_cv = torch.tensor(duration_cv,device = self.device,dtype = torch.float,requires_grad=False)
            length_cv = torch.tensor(length_cv,device = self.device,dtype = torch.float,requires_grad=False)
            return duration_cv,length_cv
        else:
            return None,None