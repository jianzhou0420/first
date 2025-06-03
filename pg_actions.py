from copy import deepcopy
import open3d as o3d
import matplotlib.pyplot as plt
import h5py

import numpy as np
from codebase.z_utils.Rotation import *
from zero.FrankaPandaFK_torch import FrankaEmikaPanda_torch
from zero.FrankaPandaFK import FrankaEmikaPanda
import torch


def compare_two_trajectories(action0, action1, label0='action0', label1='action1'):
    x = list(range(len(action0)))  # assumes all vectors have the same length
    B, D = action0.shape
    assert action1.shape == (B, D), f"action1 shape {action1.shape} does not match action0 shape {action0.shape}"

    fig, axes = plt.subplots(1, D, figsize=(12, 4))
    for i in range(D):
        axes[i].plot(x, action0[:, i], marker='o', label=label0)
        axes[i].plot(x, action1[:, i], marker='o', label=label1)
        axes[i].set_title(f'Action Dimension {i}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True)
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

        T_eePose_sim = PosQuat2HT(eePose[i])
        eePose_mesh.transform(T_eePose_sim)

        # o3d.visualization.draw_geometries([*bbox, *other_bbox, world_origin, eePose_mesh], window_name="bbox", width=1920, height=1080)

        eePose_base = franka.theta2eePose(JP[i])

        T_eePose_base = PosQuat2HT(eePose_base)

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
        T_eePose_franka = PosQuat2HT(eePose_franka) @ T_offset
        T_eePose_sim = PosQuat2HT(eePose[i])

        world_origin_o3dframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eePose_sim_o3dframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eePose_franka_03dframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        eePose_sim_o3dframe.transform(T_eePose_sim)
        eePose_franka_03dframe.transform(T_eePose_franka)

        o3d.visualization.draw_geometries([*obbox, *other_bbox, world_origin_o3dframe, eePose_sim_o3dframe, eePose_franka_03dframe], window_name="bbox", width=1920, height=1080)


def show_eePose_action_with_obs(path):
    # test1 = '/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5'
    # test1 = '/media/jian/ssd4t/equidiff/data/robomimic/datasets/stack_d1/stack_d1.hdf5'
    # test1 = '/media/jian/ssd4t/equidiff/data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5'
    np.set_printoptions(precision=3, suppress=True)
    with h5py.File(path, 'r') as f:
        data = f['data']
        demo0 = data['demo_49']
        actions = demo0['actions'][...]
        obs_ = demo0["obs"]
        ee_pos = obs_['robot0_eef_pos'][...]
        ee_quat = obs_['robot0_eef_quat'][...]
        JP = obs_['robot0_joint_pos'][...]

        eePose = np.concatenate([ee_pos, ee_quat], axis=-1)
        # print(eePose[:10])
        # print(JP[:10])
        '''
        # show_trajectory(ee_pos, ee_quat, JP)
        # find_out_T_base_and_offset(eePose, JP)
        # verify_T_base_and_offset(eePose, JP)
        '''

        # 统一向前进一格，第一格舍弃，最后一格用action补上。
        actions_eePose = eePose.copy()[1:]
        # 最后一格
        action_last = actions[-1, :]
        action_last_pos = action_last[:3]
        action_last_rot = axis2quat(action_last[:3])
        action_last = np.concatenate([action_last_pos, action_last_rot])

        actions_eePose = np.concatenate([actions_eePose, action_last[None, :]], axis=0)

        print('actions_eePose', actions_eePose[:10])
        print('eePose', eePose[:10])


def validate_JP_actions(path='/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP.hdf5'):
    T_base_mimicgen = torch.tensor([[1., 0, 0, -0.561],
                                    [0, 1., 0., 0],
                                    [0, 0., 1., 0.925],
                                    [0., 0., 0., 1.]])

    T_offset = torch.tensor([[-1, 0, 0., 0],
                             [0, -1., 0., 0],
                             [0, 0, 1., 0],
                             [0, 0, 0., 1.]])

    franka = FrankaEmikaPanda_torch()
    franka.set_T_base(T_base_mimicgen)
    franka.set_T_offset(T_offset)

    with h5py.File(path, 'r') as f:
        demo0 = f['data/demo_0']
        actions = deepcopy(demo0['actions'][...])
        JP_actions = actions[:, :-1]
        JP_obs = deepcopy(demo0['obs']['robot0_joint_pos'][...])
        pos_ee = deepcopy(demo0['obs']['robot0_eef_pos'][...])
        quat_ee = deepcopy(demo0['obs']['robot0_eef_quat'][...])

        eePose_obs = np.concatenate([pos_ee, quat_ee], axis=-1)
        eePose_obs = np.concatenate([eePose_obs[1:, :], eePose_obs[-1:, :]], axis=0)

        eePose_franka = []

        for i in range(JP_actions.shape[0]):
            theta = torch.from_numpy(JP_actions[i]).to(torch.float32)
            T_JP_actions = franka.theta2HT(theta, apply_offset=True)
            eePose_JP_actions = HT2PosQuat(T_JP_actions)
            eePose_franka.append(eePose_JP_actions)

        eePose_franka = np.array(eePose_franka)

        compare_two_trajectories(eePose_obs, eePose_franka, 'eePose_obs', 'eePose_franka')


def validate_JP_x0loss(path='data/robomimic/datasets/stack_d1/stack_d1_abs_JP_x0loss.hdf5'):
    T_base_mimicgen = torch.tensor([[1., 0, 0, -0.561],
                                    [0, 1., 0., 0],
                                    [0, 0., 1., 0.925],
                                    [0., 0., 0., 1.]])

    T_offset = torch.tensor([[-1, 0, 0., 0],
                             [0, -1., 0., 0],
                             [0, 0, 1., 0],
                             [0, 0, 0., 1.]])

    franka = FrankaEmikaPanda_torch()
    franka.set_T_base(T_base_mimicgen)
    franka.set_T_offset(T_offset)

    with h5py.File(path, 'r') as f:
        demo0 = f['data/demo_0']
        actions = deepcopy(demo0['actions'][...])
        JP_actions = actions[:, :-1]
        pos_ee = deepcopy(demo0['obs']['robot0_eef_pos'][...])
        quat_ee = deepcopy(demo0['obs']['robot0_eef_quat'][...])

        eePose_obs = np.concatenate([pos_ee, quat_ee], axis=-1)

        eePose_franka = []

        for i in range(JP_actions.shape[0]):
            theta = torch.from_numpy(JP_actions[i]).to(torch.float32)

            T_JP_actions = franka.theta2HT(theta, apply_offset=True)

            eePose_JP_actions = HT2PosQuat(T_JP_actions)
            eePose_franka.append(eePose_JP_actions)

        eePose_franka = np.array(eePose_franka)

        compare_two_trajectories(eePose_obs, eePose_franka)


if __name__ == '__main__':
    # show_eePose_action_with_obs('/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_voxel_abs_test.hdf5')
    validate_JP_actions('/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP.hdf5')
    pass
