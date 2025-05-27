from zero.FrankaPandaFK_torch import FrankaEmikaPanda_torch

import torch
import h5py
from zero.FrankaPandaFK_torch import FrankaEmikaPanda_torch
import numpy as np
from codebase.z_utils.Rotation_torch import PosEuler2HT, eePose2HT, matrix_to_rotation_6d
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)


PosEuler_base_mimicgen = torch.tensor([-0.561, 0., 0.925, 0., 0., 0.])
PosEuler_offset = torch.tensor([0., 0., 0., 0., 0., - 180.])
T_base_mimicgen = PosEuler2HT(PosEuler_base_mimicgen[None, ...])[0]
T_offset = PosEuler2HT(PosEuler_offset[None, ...])[0]

franka = FrankaEmikaPanda_torch()
franka.set_T_base(T_base_mimicgen)
franka.set_T_offset(T_offset)

with h5py.File("/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP_x0loss.hdf5", "r") as f:
    for key in f['data'].keys():
        demo = f['data'][key]
        JP = torch.tensor(demo['actions']).float().clone()[:, :7]
        eePose = torch.tensor(demo['x0loss']['eePose']).float().clone()[:, :7]
        print(f"Processing {key} with JP shape {JP.shape} and eePose shape {eePose.shape}")
        open_ = torch.tensor(demo['actions'])[:, -1:]

        T_pred = franka.theta2HT(JP, apply_offset=True)
        pos_pred = T_pred[..., :3, 3]
        mat_pred = T_pred[..., :3, :3]
        orthod6d_pred = matrix_to_rotation_6d(mat_pred)
        PosOrthod6d_pred = torch.cat([pos_pred, orthod6d_pred, open_], dim=-1)

        T_GT = eePose2HT(eePose)
        pos_GT = T_GT[..., :3, 3]
        mat_GT = T_GT[..., :3, :3]
        orthod6d_GT = matrix_to_rotation_6d(mat_GT)
        PosOrthod6d_GT = torch.cat([pos_GT, orthod6d_GT, open_], dim=-1)

        for i in range(PosOrthod6d_pred.shape[0]):
            # print(f"Frame {i}:")
            # print("Predicted:", PosOrthod6d_pred[i])
            # print("Ground Truth:", PosOrthod6d_GT[i])
            print("Difference:", PosOrthod6d_pred[i] - PosOrthod6d_GT[i])
        break
