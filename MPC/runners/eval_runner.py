from MPC.MPC_Controller.Parameters import Parameters
from MPC.MPC_Controller.robot_runner.RobotRunnerFSM import RobotRunnerFSM
from MPC.MPC_Controller.robot_runner.RobotRunnerMin import RobotRunnerMin
from MPC.MPC_Controller.robot_runner.RobotRunnerPolicy import RobotRunnerPolicy
from MPC.MPC_Controller.common.Quadruped import RobotType
from MPC.MPC_Controller.utils import DTYPE, ControllerType
from isaacgym import gymapi
from MPC.RL_Environment.sim_utils import * 
from utils.torch_utils import VecEnv,class_to_dict,dump_info,NumpyEncoder
from MPC.configs.training_config import RunnerCfg
import torch 
import numpy as np 

class Runner:
    def __init__(self, env:VecEnv,log_dir,cfg:RunnerCfg, device='cpu'):
        self.device = device
        self.env = env
        self.train_cfg = cfg 
        self.policy_cfg = cfg.policy
        self.cfg = cfg.runner
        self.ppo_cfg = cfg.algorithm

        self.num_envs = self.env.num_envs
        self.dt = Parameters.controller_dt
        self.controllers = [] 
        for _ in range(self.num_envs):
            robotRunner = RobotRunnerMin()
            robotRunner.init(RobotType.GO1)
            self.controllers.append(robotRunner)

        self.env.reset()
    
    def eval(self,num_steps,cmd = [0.5,0.0,0.0]):
        commands = np.array(cmd,dtype=DTYPE) 
        for i in range(num_steps):
            list_dof_states,list_body_states = self.env.get_mpc_observation()
            list_torques = []
            for idx,(dof_states,body_states,controller) in enumerate(zip(list_dof_states,list_body_states,self.controllers)):
                legTorques = controller.run(dof_states,body_states,commands)
                list_torques.append(legTorques / (Parameters.controller_dt*100))
            list_torques = np.stack(list_torques,axis=0)
            list_torques = torch.from_numpy(list_torques).to(self.device) 
            self.env.step(list_torques)
        print("Done")
        return 



    
    
    