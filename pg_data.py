from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
import matplotlib.pyplot as plt
import h5py

import numpy as np
from codebase.z_utils.Rotation import *
from zero.FrankaPandaFK import FrankaEmikaPanda
from zero.z_utils.h5_utils import HDF5Inspector
import torch

import json
from termcolor import cprint


def copy2new_h5py_file(src_path, dst_path):
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for name in src:
            src.copy(name, dst, name)


PosEuler_offset_action2obs = np.array([0, 0, 0, 0, 0, 90])
PosEuler_base_mimicgen = np.array([-0.561, 0., 0.925, 0., 0., 0.])
PosEuler_offset = np.array([0., 0., 0., 0., 0., - 180.])


def convert_action_to_traj(new_h5py_file):
    """
    last action is 
    """
    with h5py.File(new_h5py_file, 'r+') as f:
        data = f['data']
        print(data.keys())
        for i, key in enumerate(data.keys()):
            # 1. get a numpy copy of the dataset
            demo_data = data[key]

            PosAxisOpen_old = deepcopy(demo_data['actions'][...])  # PosAxis
            robot_ee_pos = deepcopy(demo_data['obs']['robot0_eef_pos'][...])
            robot_ee_quat = deepcopy(demo_data['obs']['robot0_eef_quat'][...])

            isOpen = PosAxisOpen_old[:, -1:]  # open action at t

            PosQuat_curr = np.concatenate([robot_ee_pos, robot_ee_quat], axis=-1)  # eePose at t

            T_curr = PosQuat2HT(PosQuat_curr)  @ inv(PosEuler2HT(PosEuler_offset_action2obs[None, ...]))

            PosAxis_new = np.concatenate((HT2PosAxis(T_curr)[1:, :], PosAxisOpen_old[-1:, :-1]), axis=0)
            PosAxisOpen_new = np.concatenate((PosAxis_new, isOpen), axis=-1)
            assert PosAxisOpen_new.shape == PosAxisOpen_old.shape, "PosAxisOpen_new shape is not equal to PosAxisOpen_old shape"

            demo_data['actions'][...] = PosAxisOpen_new
            assert np.all(demo_data['actions'][...] == PosAxisOpen_new), "demo_data['actions'] is not equal to PosAxisOpen_new"


def show_single_traj_from_h5py(path):
    with h5py.File(path, 'r') as f:
        demo0 = f['data/demo_0']
        actions = deepcopy(demo0['actions'][...])
        x = list(range(len(actions)))  # assumes all vectors have the same length

        act_dim = actions.shape[-1]
        # Plotting
        fig, axes = plt.subplots(1, act_dim, figsize=(20, 4))

        # Plot each vector in its own subplot
        for i in range(act_dim):
            axes[i].plot(x, actions[:, i], marker='o', label='actions')
            axes[i].set_title(f'ActionDim {i+1}')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()
    pass


def add_x0loss_eePose(new_h5py_file):
    with h5py.File(new_h5py_file, 'r+') as f:
        data = f['data']
        for i, key in tqdm(enumerate(data.keys())):
            demo_i_group = data[key]
            ee_pos = deepcopy(demo_i_group['obs']['robot0_eef_pos'][...])
            ee_quat = deepcopy(demo_i_group['obs']['robot0_eef_quat'][...])
            open_ = deepcopy(demo_i_group['actions'][..., -1:])

            eePose = np.concatenate((ee_pos, ee_quat), axis=-1)
            eePose = np.concatenate((eePose[1:, :], eePose[-1:, :]), axis=0)
            eePose_with_open = np.concatenate((eePose, open_), axis=-1)

            x0loss_group = demo_i_group.require_group('x0loss')
            x0loss_group.create_dataset('eePose', data=eePose_with_open, dtype='f')


def inspect_h5py_file(path):
    HDF5Inspector.inspect_hdf5(path)


def create_x0loss_from_JP(src: str):
    dst = src.replace('.hdf5', '_x0loss.hdf5')
    copy2new_h5py_file(src, dst)
    add_x0loss_eePose(dst)


def workspace():
    inspect_h5py_file('/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP_x0loss.hdf5')


if __name__ == '__main__':
    workspace()
