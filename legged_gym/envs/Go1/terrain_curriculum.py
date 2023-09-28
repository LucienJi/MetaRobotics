from legged_gym.envs.Go1.legged_robot import LeggedRobot
from legged_gym.envs.Go1.basic_config import BasicCfg  
from legged_gym.utils.terrain_v2 import Customized_Terrain
import numpy as np
import torch

"""
1. Terrain Curriculum
2. Independant of cmd 
"""

class TerrainCurriculum:
    """game-inspired curriculum.
    """
    def __init__(self,env:LeggedRobot,terrain:Customized_Terrain, cfg:BasicCfg,device) -> None:
        self.max_terrain_level = cfg.terrain.num_rows -1 
        self.max_terrain_type = cfg.terrain.num_cols - 1
        self.num_envs = env.num_envs
        self.device = device 
        self.terrain = terrain
        self.env = env 

        #! Track Each agent's terrain type and level
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)

        #! keep terrain_origins = (n_row, n_col, 3)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

        #! init terrain level and type 
        self.terrain_levels = torch.randint(0, self.max_terrain_level+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
    
    def update_curriculum(self,env_ids):
        if self.max_terrain_level == 0:
            return 
        "env_ids 已经准备 reset 了, 需要检查目前和 env ids 距离有多远"
        # print("Terrain Curriculum Debug: ")
        distance = torch.norm(self.env.root_states[env_ids, :2] - self.env.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.env.commands[env_ids, :2], dim=1)*self.env.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        # print("distance: ", distance,  
        #       "Terrain Length: ", self.terrain.env_length,
        #       "move_up: ", move_up, 
        #       "Target Vel Norm: ", torch.norm(self.env.commands[env_ids, :2], dim=1),
        #       "Max Steps: ", self.env.max_episode_length_s,
        #       "move_down", move_down)
        
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level), #! if > 0, 随机一个 level  
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)  #! else, 最差也是 0
        # print("New Levels: ", self.terrain_levels[env_ids])

    def sample_terrain(self,env_ids):
        return self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
