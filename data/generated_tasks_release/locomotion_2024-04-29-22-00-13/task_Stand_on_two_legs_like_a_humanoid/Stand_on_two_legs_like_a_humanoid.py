
from locomotion.sim import SimpleEnv
import numpy as np
import gym
from locomotion.gpt_reward_api import *

class Stand_on_two_legs_like_a_humanoid(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
            COM_pos, COM_quat = get_robot_pose(self)
            COM_vel, COM_ang = get_robot_velocity(self)
            face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)
        
            target_height = 2 * self.COM_init_pos[2] # aiming for a height that's twice the initial height
            target_up = np.array([0, 0, 1]) # maintain up direction
            target_ang = np.array([0, 0, 0]) # no spinning
        
            alpha_height = 1.0
            alpha_up = 1.0
            alpha_ang = 0.0
        
            r_height = - alpha_height * np.linalg.norm(COM_pos[2] - target_height)
            r_up     = - alpha_up * np.linalg.norm(up_dir - target_up)
            r_ang    = - alpha_ang * np.linalg.norm(COM_ang - target_ang)
            r = r_height + r_up + r_ang
        
            r_energy = get_energy_reward(self)
            return r + r_energy
gym.register(
    id='gym-Stand_on_two_legs_like_a_humanoid-v0',
    entry_point=Stand_on_two_legs_like_a_humanoid,
)
