from zero.FrankaPandaFK_torch import FrankaEmikaPanda_torch

import torch


import h5py
from zero.FrankaPandaFK_torch import FrankaEmikaPanda_torch

PosEuler_base =
franka = FrankaEmikaPanda_torch()
franka.set_T_base(torch.eye(4, dtype=torch.float32))

with h5py.File("/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP_x0loss.hdf5", "r") as f:
    for key in f['data'].keys():
        demo = f['data'][key]
        JP = torch.tensor(demo['actions']).float().clone()
        eePose = torch.tensor(demo['x0loss']['eePose']).float().clone()
        print(f"Processing {key} with JP shape {JP.shape} and eePose shape {eePose.shape}")
