
from locomotion.sim import SimpleEnv
import numpy as np
import gym
from locomotion.gpt_reward_api import *

class Rotate_on_the_spot_turning_around(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
            COM_pos, COM_quat = get_robot_pose(self)
            COM_vel, COM_ang = get_robot_velocity(self)
            face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)
        
            target_v = np.array([0, 0, 0]) # not moving
            target_height = self.COM_init_pos[2] # maintain initial height
            target_up = np.array([0, 0, 1]) # maintain up direction
            target_ang = np.array([0, 0, 1]) # spinning around the z-axis
        
            alpha_vel = 1.0
            alpha_height = 1.0
            alpha_up = 1.0
            alpha_ang = 1.0
        
            r_vel    = - alpha_vel * np.linalg.norm(COM_vel - target_v)
            r_height = - alpha_height * np.linalg.norm(COM_pos[2] - target_height)
            r_up     = - alpha_up * np.linalg.norm(up_dir - target_up)
            r_ang    = - alpha_ang * np.linalg.norm(COM_ang - target_ang)
            r = r_vel + r_height + r_up + r_ang
        
            r_energy = get_energy_reward(self)
            return r + r_energy
gym.register(
    id='gym-Rotate_on_the_spot_turning_around-v0',
    entry_point=Rotate_on_the_spot_turning_around,
)
