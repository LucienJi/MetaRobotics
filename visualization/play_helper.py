from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils import Logger
import numpy as np 
import torch 
import os 

def play_policy(env_cfg,train_cfg, policy, env, cmd_vel = [1.5,0.0,0.0],robot_idx = 10, joint_index = 2, record = False, move_camera = False):
    logger = Logger(env.dt)
    robot_index = robot_idx # which robot is used for logging
    joint_index = joint_index # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    cmd_vel = cmd_vel
    env.set_eval()
    with torch.inference_mode():
        env.set_command(cmd_vel)
        env.reset()
    obs_dict = env.get_observations()  # TODO: check, is this correct on the first step?
    
    RECORD_FRAMES = record
    MOVE_CAMERA = move_camera   
    with torch.inference_mode():
        for i in range(3*int(env.max_episode_length)):
            
            actions = policy.act_inference(obs_dict)
            obs_dict, rewards, dones, infos= env.step(actions.detach())
            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos': env.dof_pos[robot_index].cpu().numpy(),
                        'dof_pos_target': env.joint_pos_target[robot_index].cpu().numpy(),
                        'dof_vel': env.dof_vel[robot_index].cpu().numpy(),
                        'dof_torque': env.torques[robot_index].cpu().numpy(),
                        'command': env.commands[robot_index].cpu().numpy(),
                        'base_vel': env.base_lin_vel[robot_index].cpu().numpy(),
                        'base_quat': env.base_quat[robot_index].cpu().numpy(),
                        'base_ang_vel': env.base_ang_vel[robot_index].cpu().numpy(),
                        'foot_contact': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                        'action': actions[robot_index].cpu().numpy(),
                    }
                )
            elif i==stop_state_log:
                logger.new_plot_states()
                print("plotting states")
            if  0 < i < stop_rew_log:
                if infos["train/episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["train/episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()