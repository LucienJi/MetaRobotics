from MPC.MPC_Controller.common.Quadruped import Quadruped
from MPC.MPC_Controller.common.LegController import LegController
# from MPC_Controller.Parameters import Parameters
from MPC.MPC_Controller.common.StateEstimator import StateEstimator
from MPC.MPC_Controller.common.DesiredStateCommand import DesiredStateCommand

class ControlFSMData:
    def __init__(self):
        self._quadruped:Quadruped = None
        self._stateEstimator:StateEstimator = None
        self._legController:LegController = None
        self._desiredStateCommand:DesiredStateCommand = None
        # self._gaitScheduler:GaitScheduler = None
        # self.userParameters:Parameters = None

