
from locomotion.sim import SimpleEnv
import numpy as np
import gym
from locomotion.gpt_reward_api import *

class Turn_Left(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
            COM_pos, COM_quat = get_robot_pose(self)
            COM_vel, COM_ang = get_robot_velocity(self)
            face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)
        
            target_ang = np.array([0, 0, -1]) # spin around z axis to turn left 
            target_face = np.array([0, 1, 0]) # face direction should change to left
            target_up = np.array([0, 0, 1]) # maintain up direction
        
            alpha_ang = 1.0
            alpha_face = 1.0
            alpha_up = 1.0
        
            r_ang    = - alpha_ang * np.linalg.norm(COM_ang - target_ang)
            r_face   = - alpha_face * np.linalg.norm(face_dir - target_face)
            r_up     = - alpha_up * np.linalg.norm(up_dir - target_up)
            r = r_ang + r_face + r_up
        
            r_energy = get_energy_reward(self)
            return r + r_energy
gym.register(
    id='gym-Turn_Left-v0',
    entry_point=Turn_Left,
)
