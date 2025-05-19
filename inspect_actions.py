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


def find_out_T_base_and_offset(eePose, JP):
    franka = FrankaEmikaPanda()
    from numpy.linalg import inv
    for i in range(eePose.shape[0]):
        T_Base_test = np.identity(4)

        franka.set_T_base(T_Base_test)

        # eePose_franka = franka.theta2eePose(JP[0])

        bbox, other_bbox = franka.theta2obbox(JP[i])

        world_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eePose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        T_eePose_sim = eePose2HT(eePose[i])
        eePose_mesh.transform(T_eePose_sim)

        # o3d.visualization.draw_geometries([*bbox, *other_bbox, world_origin, eePose_mesh], window_name="bbox", width=1920, height=1080)

        eePose_base = franka.theta2eePose(JP[i])

        T_eePose_base = eePose2HT(eePose_base)

        T_offset = np.identity(4)
        rot_euler = np.array([0, 0, -180])
        rot_mat = euler2mat(np.radians(rot_euler))
        T_offset[:3, :3] = rot_mat

        T_base_sim = T_eePose_sim @ inv(T_offset) @ inv(T_eePose_base)
        test = HT2PosEuler(T_base_sim)

        print('--------------\n', test)
    print('T_base', T_base_sim)
    print('T_offset', T_offset)


def verify_T_base_and_offset(eePose, JP):
    '''
    you will find three frames in the o3d visualizer, which are: 1.world frame, 2. eePose_sim frame, 3. eePose_franka frame (Calculated from JP)
    the eePose_franka frame should be the same as the eePose_sim frame, if not, you need to find out the T_base and T_offset
    '''

    T_base_mimicgen = np.array([[1., 0, 0, -0.561],
                                [0, 1., 0., 0],
                                [0, 0., 1., 0.925],
                                [0., 0., 0., 1.]])
    T_offset = np.array([[-1, 0, 0., 0],
                        [0, -1., 0., 0],
                        [0, 0, 1., 0],
                        [0, 0, 0., 1.]])
    franka = FrankaEmikaPanda()
    franka.set_T_base(T_base_mimicgen)

    for i in range(eePose.shape[0]):
        theta = JP[i]
        if len(theta) == 7:
            theta = np.hstack([theta, 1])
        T_ok, T_ok_others = franka.get_T_ok(theta)
        T_ok = T_ok @ T_offset
        obbox, other_bbox = franka.get_obbox(T_ok, T_ok_others=T_ok_others, tolerance=0.005)

        eePose_franka = franka.theta2eePose(JP[i])
        T_eePose_franka = eePose2HT(eePose_franka) @ T_offset
        T_eePose_sim = eePose2HT(eePose[i])

        world_origin_o3dframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eePose_sim_o3dframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eePose_franka_03dframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        eePose_sim_o3dframe.transform(T_eePose_sim)
        eePose_franka_03dframe.transform(T_eePose_franka)

        o3d.visualization.draw_geometries([*obbox, *other_bbox, world_origin_o3dframe, eePose_sim_o3dframe, eePose_franka_03dframe], window_name="bbox", width=1920, height=1080)


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
        print(eePose[:10])
        print(JP[:10])
    '''
    # show_trajectory(ee_pos, ee_quat, JP)
    # find_out_T_base_and_offset(eePose, JP)
    # verify_T_base_and_offset(eePose, JP)
    '''
