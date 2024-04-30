
from locomotion.sim import SimpleEnv
import numpy as np
import gym
from locomotion.gpt_reward_api import *

class Crouch(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
            COM_pos, COM_quat = get_robot_pose(self)
            COM_vel, COM_ang = get_robot_velocity(self)
        
            target_height = self.COM_init_pos[2] * 0.5 # half of original height
            alpha_height = 1.0
            r_height = - alpha_height * np.linalg.norm(COM_pos[2] - target_height)
            r = r_height
        
            r_energy = get_energy_reward(self)
            return r + r_energy
gym.register(
    id='gym-Crouch-v0',
    entry_point=Crouch,
)
