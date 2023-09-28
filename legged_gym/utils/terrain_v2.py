# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from isaacgym.terrain_utils import *
from legged_gym.envs.Go1.basic_config import BasicCfg
from .terrain_lib import SubTerrain, TerrainLib

Tmpcfg = {
    'random_uniform_terrain': {
        'min_height':-0.02,
        'max_height':0.02,
        'step':0.005
    },
   
}

class Customized_Terrain:
    def __init__(self,cfg:BasicCfg.terrain
                 ) -> None:
        self.cfg = cfg 
        self.type = cfg.mesh_type
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.slope_treshold = cfg.slope_treshold
        self.num_cols = cfg.num_cols
        self.num_rows = cfg.num_rows
        self.num_sub_terrains = self.num_rows * self.num_cols
        self.env_origins = np.zeros((self.num_rows, self.num_cols, 3))
        self.horizontal_scale = cfg.horizontal_scale
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.vertical_scale = cfg.vertical_scale
        self.border_size = cfg.border_size
        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.num_rows * self.length_per_env_pixels) + 2 * self.border
        #! 地形矩阵
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.heightsamples = self.height_field_raw
        
        #! terrain kwargs
        self.terrain_cfg = cfg.terrain_kwargs 
        
        if self.type == 'plane':
            self._default_plane() 
            return 
        if cfg.selected:
            self.selected()   
        elif cfg.curriculum:
            self.curiculum() 
        else:
            self._default_plane()
            



        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.horizontal_scale,
                                                                                            self.vertical_scale,
                                                                                            self.slope_treshold)
    
        
    def curiculum(self):
        terrain_lib = TerrainLib()
        terrain_lib.parse_terrain_cfg(self.terrain_cfg)
        self.terrain_lib = terrain_lib
        for j in range(self.num_cols):
            for i in range(self.num_rows):
                difficulty = (i + 1) / self.num_rows
                choice = j / self.num_cols + 0.001
                terrain = terrain_lib.make_terrain(choice, difficulty,
                                                   name="terrain_{}_{}".format(i, j),width=self.width_per_env_pixels,
                                                   length=self.width_per_env_pixels,
                                                   vertical_scale=self.vertical_scale,
                                                   horizontal_scale=self.horizontal_scale)
                self.add_terrain_to_map(terrain, i, j)

    
    def selected(self):
        terrain_lib = TerrainLib()
        terrain_lib.parse_terrain_cfg(self.terrain_cfg)
        self.terrain_lib = terrain_lib
        for j in range(self.num_cols):
            for i in range(self.num_rows):
                difficulty = (i + 1) / self.num_rows
                terrain = terrain_lib.make_terrain(choice = None, difficulty = difficulty,
                                                   name="terrain_{}_{}".format(i, j),width=self.width_per_env_pixels,
                                                   length=self.width_per_env_pixels,
                                                   vertical_scale=self.vertical_scale,
                                                   horizontal_scale=self.horizontal_scale)
                self.add_terrain_to_map(terrain, i, j)

    def add_terrain_to_map(self, terrain:SubTerrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # NOTE: 获取原点
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def _default_plane(self):
        for j in range(self.num_cols):
            for i in range(self.num_rows):
                terrain = SubTerrain(   "terrain",width=self.width_per_env_pixels,
                                        length=self.width_per_env_pixels,
                                        vertical_scale=self.vertical_scale,
                                        horizontal_scale=self.horizontal_scale)
                self.add_terrain_to_map(terrain, i, j)