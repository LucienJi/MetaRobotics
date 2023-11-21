import numpy as np
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from .legged_robot import LeggedRobot
import torch


class RewardLib:
    def __init__(self, env: LeggedRobot):
        self.env = env

    def load_env(self, env: LeggedRobot):
        self.env = env

    # ------------ reward functions----------------
    """"Basic Reward Function"""
    # region 
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        delta = (self.env.last_dof_vel - self.env.dof_vel) / self.env.dt
        torch.where(torch.abs(delta) < 10, torch.zeros_like(delta), torch.square(delta))
        return torch.sum(torch.square(delta), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.env.dof_vel) - self.env.dof_vel_limits*self.env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        # clip to max error = 1 Nm per joint to avoid huge penalties
        return torch.sum((torch.abs(self.env.torques) - self.env.torque_limits*self.env.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)
    
    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        # self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0
        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))
        return torch.sum(rew_foot_impact_vel, dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)
    
    def _reward_joint_power(self):
        r = torch.clamp(self.env.dof_vel * self.env.torques,min=0.0)
        r = torch.sum(r, dim=1)
        return r
    
    def _reward_hip_rotate(self):
        # penalize hip rotation
        return torch.sum(torch.abs(self.env.dof_pos[:, [0, 3, 6, 9]] - \
                                   self.env.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos), dim=1) * (torch.norm(self.env.commands[:, :2], dim=1) < 0.1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts) 
        #! 这里使用 2 步 filtered 的结果
        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt
        rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.env.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.env.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stable_stride(self):
        if not hasattr(self,'_reward_stable_stride_helper'):
            self._reward_stable_stride_helper = StrideRewardHelper(self.env.num_envs,self.env.feet_indices,self.env.device)
        
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_and(contact, self.env.last_contacts) 

        first_contact = (self._reward_stable_stride_helper.feet_in_air > 0.) * contact_filt
        self._reward_stable_stride_helper.feet_in_air += self.env.dt 
        # update time 
        self._reward_stable_stride_helper.tmp_contact_time += self.env.dt * contact_filt 
        self._reward_stable_stride_helper.tmp_air_time += self.env.dt * (~contact_filt)

        # half stride complete 
        self._reward_stable_stride_helper.air_time =  self._reward_stable_stride_helper.tmp_air_time * first_contact + self._reward_stable_stride_helper.air_time * (~first_contact)
        self._reward_stable_stride_helper.contact_time = self._reward_stable_stride_helper.tmp_contact_time * first_contact + self._reward_stable_stride_helper.contact_time * (~first_contact) 

        # reset tmp, important to test the 
        # first_time = True 意味着， 已经经过了 (contact, air), 马上又要经过 (contact, air ) 
        self._reward_stable_stride_helper.tmp_air_time = torch.zeros_like(first_contact) * first_contact + self._reward_stable_stride_helper.tmp_air_time * (~first_contact)
        self._reward_stable_stride_helper.tmp_contact_time = torch.zeros_like(first_contact) * first_contact + self._reward_stable_stride_helper.tmp_contact_time * (~first_contact)
        self._reward_stable_stride_helper.feet_in_air *= ~contact_filt

        # calculate reward 
        period_cv = self._reward_stable_stride_helper._compute_period_ratio() 
        return period_cv 
    def _reward_body_height(self):
        reference_heights = 0 
        body_height = self.env.base_pos[:, 2] - reference_heights
        height_target = self.env.cfg.rewards.base_height_target - reference_heights
        return torch.square(body_height - height_target)
    
    def _reward_body_height_v2(self):
        reference_heights = 0 
        body_height = self.env.base_pos[:, 2] - reference_heights
        height_target = self.env.cfg.rewards.base_height_target - reference_heights
        delta_height = (body_height - height_target).clamp(max=0.0)
        return torch.square(delta_height)
    # endregion

    """ Custom Task Reward Functions """
    # region
    def _reward_feet_clearance(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
        target_height = self.env.foot_height.unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        
        desired_stance_width = 0.3
        desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.freq
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    def _set_freq(self,env_ids,freq):
        self.env.freq[env_ids] = freq[env_ids]
    def _set_phases(self,env_ids,phase):
        self.env.offsets[env_ids] = phase[env_ids]
        self.env.bounds[env_ids] = phase[env_ids]
    def _set_foot_height(self,env_ids,height):
        
        self.env.foot_height[env_ids] = height[env_ids]
    # endregion


    """ Task Reward Functions"""
    # region 
    def _reward_jump(self):
        reference_heights = 0
        body_height = self.env.base_pos[:, 2] - reference_heights
        jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward +=  (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4


    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.env.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    # endregion



class StrideRewardHelper:
    def __init__(self, num_envs, feet_indices, device) -> None:
        self.feet_in_air = torch.zeros((num_envs, len(feet_indices)),device = device) #! utils

        self.tmp_contact_time = torch.zeros((num_envs, len(feet_indices)),device = device)
        self.tmp_air_time = torch.zeros((num_envs,len(feet_indices)), device = device)
        self.tmp_stride_length = torch.zeros((num_envs,len(feet_indices)), device = device)

        self.contact_time = torch.ones((num_envs, len(feet_indices)),device = device)
        self.air_time = torch.randn((num_envs,len(feet_indices)), device = device).abs()
        self.stride_length = torch.randn((num_envs,len(feet_indices)), device = device).abs()
    
    def _compute_period_ratio(self):
        ratio = self.air_time / (self.contact_time + 1e-6) # num_envs, 4 
        r_mean = ratio.mean(-1)
        r_std = ratio.std(-1)
        cv = r_std / (r_mean + 1e-6)
        return cv 
    def _compute_length_ratio(self):
        l_mean = self.stride_length.mean(-1)
        l_std = self.stride_length.std(-1)
        cv = l_std/ (l_mean + 1e-6)
        return cv 