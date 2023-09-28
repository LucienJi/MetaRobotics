import time
import numpy as np
from MPC.MPC_Controller.common.DesiredStateCommand import DesiredStateCommand
from MPC.MPC_Controller.FSM_states.ControlFSM import ControlFSM
from MPC.MPC_Controller.FSM_states.ControlFSMData import ControlFSMData
from MPC.MPC_Controller.Parameters import Parameters
from MPC.MPC_Controller.common.Quadruped import Quadruped, RobotType
from MPC.MPC_Controller.common.LegController import LegController
from MPC.MPC_Controller.common.StateEstimator import StateEstimator
from MPC.MPC_Controller.convex_MPC.ConvexMPCLocomotion import ConvexMPCLocomotion
from MPC.MPC_Controller.utils import DTYPE
from MPC.RL_Environment.WeightPolicy import WeightPolicy
from MPC.RL_Environment.utils.utils import set_np_formatting

class RobotRunnerPolicy:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

    def init(self, robotType:RobotType):
        """
        Initializes the robot model, state estimator, leg controller,
        robot data, and any control logic specific data.
        """
        self.robotType = robotType
        task = robotType.name.title()
        if self.checkpoint is None:
            self.checkpoint = f"RL_Environment/runs/{task}/nn/{task}.pth"

        # init quadruped
        if self.robotType in RobotType:
            self._quadruped = Quadruped(self.robotType)
        else:
            raise Exception("Invalid RobotType")

        # init leg controller
        self._legController = LegController(self._quadruped)

        # init state estimator
        self._stateEstimator = StateEstimator(self._quadruped)

        # init desired state command
        self._desiredStateCommand = DesiredStateCommand()

        # init weight policy
        self._weightPolicy = WeightPolicy(task=task,
                                          checkpoint=self.checkpoint)
        weights = self._quadruped._mpc_weights[:-1] # keep weights shape (12,)
        # weights.pop() # keep weights shape (12,)
        # self.weights = np.asarray(weights, dtype=DTYPE)
        self.weights = weights

        # Controller initializations
        self._controlFSM = ControlFSM(self._quadruped, 
                                      self._stateEstimator, 
                                      self._legController,
                                      self._desiredStateCommand)
        # set_np_formatting()

    def reset(self):
        self._controlFSM.initialize()

    def run(self, dof_states, body_states, commands):
        """
        Runs the overall robot control system by calling each of the major components
        to run each of their respective steps.
        """

        # Update the joint states
        self._legController.updateData(dof_states)
        self._legController.zeroCommand()
        # self._legController.setEnable(True)

        # Update robot states
        self._stateEstimator.update(body_states)

        # step weight policy
        timer = time.time()
        self._weightPolicy.compute_observations(dof_states, 
                                                self._stateEstimator.getResult(), 
                                                commands, 
                                                self.weights)
        self.weights = self._weightPolicy.step() # shape (12,)
        if Parameters.policy_print_time:
            print("Policy Update Time: {:.5f}".format(time.time()-timer))
        # print("MPC Weights:")
        # print(self.weights)

        # Update desired commands
        self._desiredStateCommand.updateCommand(commands, self.weights)

        # Run the Control FSM code
        self._controlFSM.runFSM()

        # Sets the leg controller commands for the robot
        legTorques = self._legController.updateCommand()

        return legTorques # numpy (12,) float32