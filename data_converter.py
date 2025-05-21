from copy import deepcopy
import open3d as o3d
import matplotlib.pyplot as plt
import h5py

import numpy as np
from codebase.z_utils.Rotation import *
from zero.FrankaPandaFK import FrankaEmikaPanda

import torch


def show_trajectory(ee_pos, actions):
    # X axis = list indices
    x = list(range(len(ee_pos)))  # assumes all vectors have the same length

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot each vector in its own subplot
    axes[0].plot(x, ee_pos[:, 0], marker='o', label='ee_pos')
    axes[0].plot(x, actions[:, 0], marker='o', label='actions')
    axes[0].set_title('X')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x, ee_pos[:, 1], marker='o', label='ee_pos')
    axes[1].plot(x, actions[:, 1], marker='o', label='actions')
    axes[1].set_title('Y')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(x, ee_pos[:, 2], marker='o', label='ee_pos')
    axes[2].plot(x, actions[:, 2], marker='o', label='actions')
    axes[2].set_title('Z')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def copy2new_h5py_file(src_path, dst_path):
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for name in src:
            src.copy(name, dst, name)


PosEuler_offset_action2obs = np.array([0, 0, 0, 0, 0, 90])
PosEuler_base_mimicgen = np.array([-0.561, 0., 0.925, 0., 0., 0.])
PosEuler_offset = np.array([0., 0., 0., 0., 0., - 180.])


def convert_action_to_traj(new_h5py_file):

    with h5py.File(new_h5py_file, 'r+') as f:
        data = f['data']
        print(data.keys())
        for i, key in enumerate(data.keys()):
            # 1. get a numpy copy of the dataset
            demo_data = data[key]
            print(data['demo_0'].keys())  # <KeysViewHDF5 ['actions', 'dones', 'obs', 'rewards', 'states']>

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


def workspace():
    h5py_file = '/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP.hdf5'
    np.set_printoptions(precision=3, suppress=True)

    with h5py.File(h5py_file, 'r') as f:
        data = f['data']
        print(data.keys())
        for i, key in enumerate(data.keys()):
            # 1. get a numpy copy of the dataset
            demo_data = data[key]
            print(data['demo_0'].keys())  # <KeysViewHDF5 ['actions', 'dones', 'obs', 'rewards', 'states']>

            PosAxisOpen_oldaction = deepcopy(demo_data['actions'][...])  # PosAxis
            JP_curr = deepcopy(demo_data['obs']['robot0_joint_pos'][...])
            isOpen = PosAxisOpen_oldaction[:, -1:]  # open action at t
            JP_new = np.concatenate((JP_curr[1:, :], JP_curr[-1:, :]), axis=0)  # 最后一个弄两遍。
            JPOpen_newaction = np.concatenate((JP_new, isOpen), axis=-1)
            demo_data['actions'][...] = JPOpen_newaction


if __name__ == '__main__':
    workspace()
