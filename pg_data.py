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


def change_controller_type(path):
    with h5py.File(path, 'r+') as f:
        data = f['data']
        env_meta = json.loads(f["data"].attrs["env_args"])
        # for i, key in enumerate(data.keys()):
        #     # 1. get a numpy copy of the dataset
        #     demo_data = data[key]
        #     print(data['demo_0'].keys())
        cprint(env_meta, 'blue')
        new_env_meata = {"env_name": "Stack_D1",
                         "env_version": "1.4.1",
                         "type": 1,
                         "env_kwargs": {"has_renderer": False,
                                        "has_offscreen_renderer": True,
                                        "ignore_done": True,
                                        "use_object_obs": True,
                                        "use_camera_obs": True,
                                        "control_freq": 20,
                                        "controller_configs": {"type": "JOINT_POSITION",
                                                               "input_max": 1,
                                                               "input_min": -1,
                                                               "output_max": 0.05,
                                                               "output_min": -0.05,
                                                               "kp": 150,
                                                               "damping_ratio": 1,
                                                               "impedance_mode": "fixed",
                                                               "kp_limits": [0, 300],
                                                               "damping_ratio_limits": [0, 10],
                                                               "qpos_limits": None,
                                                               'control_delta': True,
                                                               "interpolation": None,
                                                               "ramp_ratio": 0.2},
                                        "robots": ["Panda"],
                                        "camera_depths": False,
                                        "camera_heights": 84,
                                        "camera_widths": 84,
                                        "render_gpu_device_id": 0,
                                        "reward_shaping": False,
                                        "camera_names": ["birdview", "agentview", "sideview", "robot0_eye_in_hand"]}}
        json_str = json.dumps(new_env_meata)
        f["data"].attrs["env_args"] = json_str
        cprint('Done !Changed env_args to', 'green')
    with h5py.File(path, 'r') as f:
        env_args = f["data"].attrs["env_args"]
        cprint(env_args, 'blue')


def changeData2JP(path):

    # change_controller_type(JP_h5py_file)
    with h5py.File(path, 'r+') as f:
        data = f['data']
        for i, key in enumerate(data.keys()):
            # 1. get a numpy copy of the dataset
            demo_data = data[key]
            action_data = deepcopy(demo_data['actions'][...])
            JP_all = deepcopy(demo_data['obs']['robot0_joint_pos'][...])

            open_action = action_data[:, -1:]  # open action at t
            JP_all_new = np.concatenate((JP_all[1:, :], JP_all[-1:, :]), axis=0)
            JPOpen_all_new = np.concatenate((JP_all_new, open_action), axis=-1)
            del demo_data['actions']
            demo_data['actions'] = JPOpen_all_new

    with h5py.File(path, 'r') as f:
        data = f['data']
        for i, key in enumerate(data.keys()):
            # 1. get a numpy copy of the dataset
            demo_data = data[key]
            print(key, demo_data.keys())
            cprint(demo_data['actions'].shape, 'green')
            break


def DataseteePose2JP(path):
    new_path = path.replace('.hdf5', '_JP.hdf5')
    copy2new_h5py_file(path, new_path)
    change_controller_type(new_path)
    changeData2JP(new_path)


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
