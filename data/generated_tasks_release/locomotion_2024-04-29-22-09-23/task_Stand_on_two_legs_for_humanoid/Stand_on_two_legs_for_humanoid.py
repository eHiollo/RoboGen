
from locomotion.sim import SimpleEnv
import numpy as np
import gym
from locomotion.gpt_reward_api import *

class Stand_on_two_legs_for_humanoid(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
            COM_pos, COM_quat = get_robot_pose(self)
            COM_vel, COM_ang = get_robot_velocity(self)
            face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)
        
            target_up = np.array([0, 0, 1]) # maintain up direction
            alpha_up = 1.0
        
            # penalize the robot if it is not upright
            r_up = - alpha_up * np.linalg.norm(up_dir - target_up)
            r = r_up
        
            # penalize the robot if it is not balancing on two legs
            leg1_pos, leg2_pos = get_legs_position(self)
            alpha_balance = 1.0
            r_balance = - alpha_balance * np.linalg.norm(leg1_pos - leg2_pos)
            r += r_balance
        
            r_energy = get_energy_reward(self)
            return r + r_energy
gym.register(
    id='gym-Stand_on_two_legs_for_humanoid-v0',
    entry_point=Stand_on_two_legs_for_humanoid,
)
