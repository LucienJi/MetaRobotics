import numpy as np
from numpy.random import choice
from scipy import interpolate
from isaacgym import gymutil, gymapi
from math import sqrt

class SubTerrain:
    def __init__(self,
                 terrain_name="terrain", 
                 width=256, length=256, 
                 vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


class TerrainLib:
    def __init__(self):
        return 
    def parse_terrain_cfg(self,terrain_cfg:dict):
        self.terrain_cfg = terrain_cfg
        self.terrain_func = {}
        self.terrain_names = []
        self.terrain_weights = []
        for name, cfg in terrain_cfg.items():
            if hasattr(self,name):
                self.terrain_names.append(name)
                w = cfg.pop("weight", 1)
                self.terrain_weights.append(w)
                self.terrain_func[name] = lambda name,terrain, difficulty: getattr(self,name)(terrain, difficulty, 
                                                                                              **self.terrain_cfg[name])
            else:
                print(f"Warning : {name} is not a valid terrain function")
        self.terrain_weights = np.array(self.terrain_weights)
        self.terrain_weights /= self.terrain_weights.sum() #! 用 weight 来控制地形的比重

    def make_terrain(self,choice, difficulty,
                     name = 'terrain',
                     width=256, length=256,vertical_scale=1.0, horizontal_scale=1.0):
        terrain = SubTerrain(terrain_name=name,width = width,length = length,vertical_scale=vertical_scale,horizontal_scale=horizontal_scale)
        terrain_type = 0 
        if choice is not None:
            for i,name in enumerate(self.terrain_names):
                if choice < self.terrain_weights[:i+1].sum():
                    terrain_type = i
                    break
        else:
            terrain_type = np.argmax(self.terrain_weights)
            difficulty = 1.0 
        terrain_type = self.terrain_names[terrain_type]
        terrain = self.terrain_func[terrain_type](terrain_type,terrain, difficulty)
        return terrain
    # ---------------- Terrain Library --------------
    """Basic terrains"""
    # region 
    def plane_terrain(self,terrain:SubTerrain, difficulty, height = 0.0):
        max_height = int(height / terrain.vertical_scale)
        terrain.height_field_raw += max_height
        return terrain
    def random_uniform_terrain(self,terrain:SubTerrain, difficulty, min_height, max_height, step=1, downsampled_scale=None,):
        if downsampled_scale is None:
            downsampled_scale = terrain.horizontal_scale
        # difficulty < 1 
        difficulty = np.clip(difficulty, 0, 1) 
        min_height = min_height * difficulty
        max_height = max_height * difficulty

        # switch parameters to discrete units
        min_height = int(min_height / terrain.vertical_scale)
        max_height = int(max_height / terrain.vertical_scale)
        step = int(step / terrain.vertical_scale)

        heights_range = np.arange(min_height, max_height + step, step)
        height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
            terrain.length * terrain.horizontal_scale / downsampled_scale)))

        x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
        y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

        f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

        x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
        y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
        z_upsampled = np.rint(f(y_upsampled, x_upsampled))

        terrain.height_field_raw += z_upsampled.astype(np.int16)
        return terrain
    def sloped_terrain(self,terrain:SubTerrain, difficulty, slope=1):
        
        difficulty = np.clip(difficulty, 0, 1)
        slope = slope * difficulty
        
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width) # 单个坡
        terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
        return terrain
    def pyramid_sloped_terrain(self,terrain:SubTerrain,difficulty, slope=1, platform_size=1.):
        difficulty = np.clip(difficulty, 0, 1)
        slope = slope * difficulty 
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        center_x = int(terrain.width / 2)
        center_y = int(terrain.length / 2)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = (center_x - np.abs(center_x-xx)) / center_x
        yy = (center_y - np.abs(center_y-yy)) / center_y
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
        terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

        platform_size = int(platform_size / terrain.horizontal_scale / 2) #! 除以2是因为平台是正方形, 可以是爬上和爬下
        x1 = terrain.width // 2 - platform_size
        x2 = terrain.width // 2 + platform_size
        y1 = terrain.length // 2 - platform_size
        y2 = terrain.length // 2 + platform_size

        min_h = min(terrain.height_field_raw[x1, y1], 0)
        max_h = max(terrain.height_field_raw[x1, y1], 0)
        terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
        return terrain
    def discrete_obstacles_terrain(self,terrain:SubTerrain, difficulty,max_height, min_size = 1.0, max_size = 2.0, num_rects = 20, platform_size=1.):
        # switch parameters to discrete units
        difficulty = np.clip(difficulty, 0, 1)   

        max_height = 0.05 + difficulty * max_height
        num_rectangles =  int(difficulty * num_rects)

        max_height = int(max_height / terrain.vertical_scale)
        min_size = int(min_size / terrain.horizontal_scale)
        max_size = int(max_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        (i, j) = terrain.height_field_raw.shape
        height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
        width_range = range(min_size, max_size, 4)
        length_range = range(min_size, max_size, 4)

        for _ in range(num_rectangles):
            width = np.random.choice(width_range)
            length = np.random.choice(length_range)
            start_i = np.random.choice(range(0, i-width, 4))
            start_j = np.random.choice(range(0, j-length, 4))
            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

        x1 = (terrain.width - platform_size) // 2
        x2 = (terrain.width + platform_size) // 2
        y1 = (terrain.length - platform_size) // 2
        y2 = (terrain.length + platform_size) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0
        return terrain
    def wave_terrain(self,terrain:SubTerrain, difficulty, num_waves=1, amplitude=1.):
        difficulty = np.clip(difficulty, 0, 1)
        amplitude  = amplitude * difficulty 
        amplitude = int(0.5*amplitude / terrain.vertical_scale)
        if num_waves > 0:
            div = terrain.length / (num_waves * np.pi * 2)
            x = np.arange(0, terrain.width)
            y = np.arange(0, terrain.length)
            xx, yy = np.meshgrid(x, y, sparse=True)
            xx = xx.reshape(terrain.width, 1)
            yy = yy.reshape(1, terrain.length)
            terrain.height_field_raw += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
                terrain.height_field_raw.dtype)
        return terrain
    def stairs_terrain(self,terrain:SubTerrain,difficulty, step_width, step_height):
        # switch parameters to discrete units
        difficulty = np.clip(difficulty, 0, 1) 
        step_height = step_height * difficulty  
        # step_width = step_width * ( 1.5 - 0.5 * difficulty) #! 就是宽一点是不是好走
        step_width = step_width 
        step_width = int(step_width / terrain.horizontal_scale)
        step_height = int(step_height / terrain.vertical_scale)

        num_steps = terrain.width // step_width
        height = step_height
        for i in range(num_steps):
            terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
            height += step_height
        return terrain

    def pyramid_stairs_terrain(self,terrain:SubTerrain, difficulty,step_width, step_height, platform_size=1.):
        # switch parameters to discrete units
        difficulty = np.clip(difficulty, 0, 1) 
        step_height = 0.05 + step_height * difficulty 
        step_width =  0.35  #! 就是宽一点是不是好走

        step_width = int(step_width / terrain.horizontal_scale)
        step_height = int(step_height / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        height = 0
        start_x = 0
        stop_x = terrain.width
        start_y = 0
        stop_y = terrain.length
        while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
            start_x += step_width
            stop_x -= step_width
            start_y += step_width
            stop_y -= step_width
            height += step_height
            terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
        return terrain

    def stepping_stones_terrain(self,terrain:SubTerrain,difficulty,  stone_size, stone_distance, max_height=0, platform_size=4., depth=-10):
        difficulty = np.clip(difficulty, 0, 1)

        stone_size = stone_size * (1.5 + 1.0 - difficulty)
        stone_distance = 0.05 if difficulty==0 else stone_distance
        # switch parameters to discrete units
        stone_size = int(stone_size / terrain.horizontal_scale)
        stone_distance = int(stone_distance / terrain.horizontal_scale)
        max_height = int(max_height / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)
        height_range = np.arange(-max_height-1, max_height, step=1)

        start_x = 0
        start_y = 0
        terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
        if terrain.length >= terrain.width:
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                start_x = np.random.randint(0, stone_size)
                # fill first hole
                stop_x = max(0, start_x - stone_distance)
                terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
                # fill row
                while start_x < terrain.width:
                    stop_x = min(terrain.width, start_x + stone_size)
                    terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                    start_x += stone_size + stone_distance
                start_y += stone_size + stone_distance
        elif terrain.width > terrain.length:
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                start_y = np.random.randint(0, stone_size)
                # fill first hole
                stop_y = max(0, start_y - stone_distance)
                terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
                # fill column
                while start_y < terrain.length:
                    stop_y = min(terrain.length, start_y + stone_size)
                    terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                    start_y += stone_size + stone_distance
                start_x += stone_size + stone_distance

        x1 = (terrain.width - platform_size) // 2
        x2 = (terrain.width + platform_size) // 2
        y1 = (terrain.length - platform_size) // 2
        y2 = (terrain.length + platform_size) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0
        return terrain


    def gap_terrain(self,terrain:SubTerrain, difficulty, gap_size, platform_size=1.):
        # 要跳出 platform 
        difficulty = np.clip(difficulty, 0, 1)
        gap_size = gap_size * difficulty 
        gap_size = int(gap_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        center_x = terrain.length // 2
        center_y = terrain.width // 2
        x1 = (terrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (terrain.width - platform_size) // 2
        y2 = y1 + gap_size
    
        terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -100
        terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0
        return terrain 

    def pit_terrain(self,terrain:SubTerrain,difficulty, depth, platform_size=1.):
        # 要从 坑中爬出来
        difficulty = np.clip(difficulty, 0, 1) 
        depth = depth * difficulty 
        depth = int(depth / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        x1 = terrain.length // 2 - platform_size
        x2 = terrain.length // 2 + platform_size
        y1 = terrain.width // 2 - platform_size
        y2 = terrain.width // 2 + platform_size
        terrain.height_field_raw[x1:x2, y1:y2] = -depth
        return terrain 
    # endregion