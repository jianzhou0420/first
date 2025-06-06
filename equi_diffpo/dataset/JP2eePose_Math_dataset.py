import open3d as o3d
import pickle
import re
from torch.utils.data import Dataset
import torch
import os
import numpy as np

import time
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from codebase.z_utils.Rotation import *
import math
from zero.FrankaPandaFK import FrankaEmikaPanda
from zero.z_utils.coding import npa
from zero.z_utils.normalizer_action import normalize_pos, quat2ortho6D


# --------------------------------------------------------------
# region Dataset
JOINT_POSITION_LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]]


def normalize_JP(JP):
    lower = JOINT_POSITION_LIMITS[0]
    upper = JOINT_POSITION_LIMITS[1]
    if isinstance(JP, np.ndarray):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        # broadcasting
        normalized_JP = 2 * (JP - lower_np) / (upper_np - lower_np) - 1

    elif isinstance(JP, torch.Tensor):
        lower_tensor = torch.tensor(lower, device=JP.device)
        upper_tensor = torch.tensor(upper, device=JP.device)
        # broadcasting
        normalized_JP = 2 * (JP - lower_tensor) / (upper_tensor - lower_tensor) - 1
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return normalized_JP


def denormalize_JP(norm_JP):
    lower = JOINT_POSITION_LIMITS[0]
    upper = JOINT_POSITION_LIMITS[1]
    if isinstance(norm_JP, np.ndarray):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        # broadcasting
        JP = lower_np + (norm_JP + 1) / 2 * (upper_np - lower_np)

    elif isinstance(norm_JP, torch.Tensor):
        lower_t = torch.tensor(lower, device=norm_JP.device)
        upper_t = torch.tensor(upper, device=norm_JP.device)
        # broadcasting
        JP = lower_t + (norm_JP + 1) / 2 * (upper_t - lower_t)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return JP


class DatasetGeneral(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.franka = FrankaEmikaPanda()
        self.JOINT_POSITION_LIMITS = npa([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                                          [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]])
        self.low = self.JOINT_POSITION_LIMITS[0]
        self.high = self.JOINT_POSITION_LIMITS[1]

    def get_normalizer(self,):
        pass

    def __len__(self):
        return self.config['Dataset']['length']

    def __getitem__(self, idx):
        JP = npa(np.random.uniform(self.low, self.high, size=(1, self.low.shape[0])))
        eePose = npa([self.franka.theta2eePose(JP[i]) for i in range(JP.shape[0])]).reshape(1, 1, -1)
        PosOrtho6D = np.zeros((9))
        JP = normalize_JP(JP)

        PosOrtho6D[:3] = normalize_pos(eePose[..., :3])
        PosOrtho6D[3:] = quat2ortho6D(eePose[..., 3:])

        JP = torch.from_numpy(npa(JP).squeeze(0)).float()
        eePose = torch.from_numpy(npa(PosOrtho6D).squeeze()).float()

        if self.config['Model']['FK']:
            return {
                'input': JP,
                'output': eePose,
            }
        else:
            return {
                'input': eePose,
                'output': JP,
            }
    # endregion
