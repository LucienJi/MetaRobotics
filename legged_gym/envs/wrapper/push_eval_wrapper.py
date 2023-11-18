from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.Go1.legged_robot import LeggedRobot
import numpy as np 
import os 
import torch
import gym
from collections import defaultdict


class PushConfig:
    def __init__(self,
                 id,
                 body_index_list:list,
                 change_interval:int,
                 force_list:list, ) -> None:
        self.id = id 
        self.body_index_list =body_index_list
        self.change_interval = change_interval
        self.force_list = force_list 
        assert len(self.body_index_list) > 0 and  len(self.force_list) > 0
        self._force = self.force_list[0]
        self._body_index = self.body_index_list[0] 
    
    def _change(self):
        self._force = np.random.choice(self.force_list)
        self._body_index = np.random.choice(self.body_index_list) 

class EvalWrapper():
    def __init__(self, env:LeggedRobot, env_cfg, cmd_vel = [0.5, 0.0,0.0],
                 record = False, move_camera = False,experiment_name = 'Eval'):
        self.env = env
        self.env.set_eval()
        self.eval_config = None
        self.experiment_name = experiment_name

        self.camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        self.camera_vel = np.array([1., 1., 0.])
        self.camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        self.env.set_eval()
        with torch.inference_mode():
            self.env.reset()
            self.env.reset_force_to_apply()
            self.env.set_command(cmd_vel)
        self.obs_dict = self.env.get_observations()
        self.record = record 
        self.move_camera = move_camera

        self.step_ct = 0
        self.img_idx = 0
        self.eval_res = defaultdict(list)

    def set_eval_config(self, eval_config):
        if type(eval_config) is not list:
            self.eval_config = [eval_config]
        else:
            self.eval_config = eval_config
    
    
    def step(self, action):
        for config in self.eval_config:
            self.env.set_force_apply(config._body_index, config._force, z_force_norm = 0)
            if config.change_interval > 0 and (self.step_ct% config.change_interval == 0):
                config._change()

        self.obs_dict, rewards, dones, infos= self.env.step(action.detach())
        eval_res = self.env.get_push_data()
        for k,v in eval_res.items():
            self.eval_res[k].append(v)

        self.env.reset_force_to_apply()


        self.step_ct += 1    

        

        if self.record:
            if self.step_ct % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name, 'exported', 'frames', f"{self.img_idx}.png")
                self.env.gym.write_viewer_image_to_file(self.env.viewer, filename)
                self.img_idx += 1 
        if self.move_camera:
            self.camera_position += self.camera_vel * self.env.dt
            self.env.set_camera(self.camera_position, self.camera_position + self.camera_direction)
        
    def get_result(self):
        for k,v in self.eval_res.items():
            if type(v) == list:
                self.eval_res[k] = np.stack(v, axis=1) # (n_env, n_step)
        first_done = np.argmax(self.eval_res['done'], axis = 1)
        self.eval_res['first_done'] = first_done
        self.eval_res['Fall'] = first_done < 1000
        return self.eval_res
        # if os.path.exists(eval_path) == False:
        #     os.makedirs(eval_path)
        # eval_file_name = os.path.join(eval_path,eval_name)
        # np.save(eval_file_name, eval_res)