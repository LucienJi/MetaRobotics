from isaacgym import gymutil, gymapi
import torch

from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym.envs.Go1.quad_config import QuadCfg

class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self,sim_params, sim_device, headless,
                 cfg: QuadCfg = None, eval_cfg: QuadCfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)

    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def set_commands(self, commands):
        if not isinstance(commands, torch.Tensor): 
            commands = torch.tensor(commands, dtype=torch.float32, device=self.device)
        if commands.shape[0] != self.num_envs:
            commands = commands.reshape(1,3).repeat(self.num_envs, 1)
        self.commands[:,0:3] = commands[:,0:3].clone()

