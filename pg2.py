import numpy as np
import h5py
import open3d as o3d
import matplotlib.pyplot as plt
from numpy.linalg import inv
from codebase.z_utils.Rotation import *


def show_trajectory(ee_pos, actions):
    # X axis = list indices
    x = list(range(len(ee_pos)))  # assumes all vectors have the same length

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot each vector in its own subplot
    axes[0].plot(x[:-1], ee_pos[1:, 0], marker='o', label='ee_pos')
    axes[0].plot(x, actions[:, 0], marker='o', label='actions')
    axes[0].set_title('X')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x[:-1], ee_pos[1:, 1], marker='o', label='ee_pos')
    axes[1].plot(x, actions[:, 1], marker='o', label='actions')
    axes[1].set_title('Y')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(x[:-1], ee_pos[1:, 2], marker='o', label='ee_pos')
    axes[2].plot(x, actions[:, 2], marker='o', label='actions')
    axes[2].set_title('Z')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # test1 = '/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5'
    test1 = 'data/robomimic/datasets/stack_d1/stack_d1_voxel_abs.hdf5'
    # test1 = '/media/jian/ssd4t/equidiff/data/robomimic/datasets/stack_d1/stack_d1.hdf5'
    # test1 = '/media/jian/ssd4t/equidiff/data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5'
    np.set_printoptions(precision=3, suppress=True)
    with h5py.File(test1, 'r') as f:
        data = f['data']

        demo0 = data['demo_49']
        actions = demo0['actions'][...]
        obs_ = demo0["obs"]
        ee_pos = obs_['robot0_eef_pos'][...]
        ee_quat = obs_['robot0_eef_quat'][...]
        JP = obs_['robot0_joint_pos'][...]

        eePose = np.concatenate([ee_pos, ee_quat], axis=-1)
