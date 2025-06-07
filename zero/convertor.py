import torch

"""
original data is equidiff 
0. stack_d1_abs.hdf5

we convert it to three versions:
1. stack_d1_traj_eePose.hdf5
2. stack_d1_traj_JP.hdf5
3. stack_d1_traj_JP_eeloss.hdf5
"""
import h5py
import numpy as np
from copy import deepcopy
from codebase.z_utils.Rotation import PosQuat2HT, HT2PosAxis, PosEuler2HT, inv, axis2quat
from termcolor import cprint
import json
from tqdm import tqdm


class HDF5Inspector:
    '''
    Print the structure of an HDF5 file in a tree format,
    but limit each group to displaying only 10 child keys.
    Usage:
    HDF5Inspector.inspect_hdf5('path/to/your/file.hdf5')
    '''
    MAX_KEYS = 15  # Maximum number of child items to display per group # TODO： make this a parameter

    @staticmethod
    def print_tree(name, obj, prefix='', is_last=True):
        connector = '└── ' if is_last else '├── '
        # Print either the root or a group/dataset name
        if isinstance(obj, h5py.Group):
            if name == '/':
                print(name)
            else:
                print(f"{prefix}{connector}{name.split('/')[-1]}")
            # Print any attributes on the group
            if obj.attrs:
                for i, attr in enumerate(obj.attrs):
                    is_last_attr = (i == len(obj.attrs) - 1 and len(obj) == 0)
                    attr_connector = '└── ' if is_last_attr else '├── '
                    attr_prefix = prefix + ('    ' if is_last else '│   ')
                    print(f"{attr_prefix}{attr_connector}@{attr}  ⟵ Attribute on group")
            # Get child items, but limit to MAX_KEYS
            items = list(obj.items())
            total_children = len(items)
            items_to_print = items[:HDF5Inspector.MAX_KEYS]
            for idx, (child_name, child_obj) in enumerate(items_to_print):
                last = (idx == len(items_to_print) - 1) and (total_children <= HDF5Inspector.MAX_KEYS)
                new_prefix = prefix + ('    ' if is_last else '│   ')
                HDF5Inspector.print_tree(child_name, child_obj, new_prefix, last)
            # If there are more than MAX_KEYS children, indicate truncation
            if total_children > HDF5Inspector.MAX_KEYS:
                trunc_prefix = prefix + ('    ' if is_last else '│   ')
                print(f"{trunc_prefix}└── ... and {total_children - HDF5Inspector.MAX_KEYS} more items")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{prefix}{connector}{name.split('/')[-1]}  ⟵ Dataset (shape: {shape}, dtype: {dtype})")
            # Print any attributes on the dataset
            if obj.attrs:
                for i, attr in enumerate(obj.attrs):
                    is_last_attr = (i == len(obj.attrs) - 1)
                    attr_connector = '└── ' if is_last_attr else '├── '
                    attr_prefix = prefix + ('    ' if is_last else '│   ')
                    print(f"{attr_prefix}{attr_connector}@{attr}  ⟵ Attribute on dataset")

    @staticmethod
    def inspect_hdf5(file_path):
        """Inspect and print the structure of the given HDF5 file."""
        try:
            with h5py.File(file_path, 'r') as f:
                HDF5Inspector.print_tree('/', f['/'], '', True)
        except Exception as e:
            print(f"Error opening file '{file_path}': {e}")


class DatasetConvertor:
    '''
    receive original data stack_d1_abs.hdf5
    '''
    PosEuler_offset_action2obs = np.array([0, 0, 0, 0, 0, 90])
    PosEuler_base_mimicgen = np.array([-0.561, 0., 0.925, 0., 0., 0.])
    PosEuler_offset_JP2eePose = np.array([0., 0., 0., 0., 0., - 180.])

    def traj_eePose(self, original_path: str):
        traj_eePose_path = original_path.replace('.hdf5', '_traj_eePose.hdf5')

        cprint(f"Converting\n{original_path}\nto{traj_eePose_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, traj_eePose_path)

        with h5py.File(traj_eePose_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                PosAxisOpen_old = deepcopy(demo_data['actions'][...])  # PosAxis
                robot_ee_pos = deepcopy(demo_data['obs']['robot0_eef_pos'][...])
                robot_ee_quat = deepcopy(demo_data['obs']['robot0_eef_quat'][...])
                isOpen = PosAxisOpen_old[:, -1:]  # open action at t

                # 2. convert to eePose-PosAxis
                PosQuat_curr = np.concatenate([robot_ee_pos, robot_ee_quat], axis=-1)  # eePose at t
                PosAxis_curr = self._PosQuat2PosAxis(PosQuat_curr)

                # 3. convert to new PosAxisOpen
                PosAxis_new = np.concatenate((PosAxis_curr[1:, :], PosAxisOpen_old[-1:, :-1]), axis=0)
                PosAxisOpen_new = np.concatenate((PosAxis_new, isOpen), axis=-1)

                assert PosAxisOpen_new.shape == PosAxisOpen_old.shape, "PosAxisOpen_new shape is not equal to PosAxisOpen_old shape"

                demo_data['actions'][...] = PosAxisOpen_new
                assert np.all(demo_data['actions'][...] == PosAxisOpen_new), "demo_data['actions'] is not equal to PosAxisOpen_new"
        cprint(f"Convertion has been done\n You should find{traj_eePose_path}", 'green')

    def traj_JP(self, original_path: str):
        '''
        1. copy original_path to traj_JP_path
        2. change controller type to JOINT_POSITION
        3. convert actions from PosAxisOpen to JP
        '''

        traj_JP_path = original_path.replace('.hdf5', '_traj_JP.hdf5')
        cprint(f"Converting\n{original_path}\nto{traj_JP_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, traj_JP_path)
        self._controller_type_to_JP(traj_JP_path)

        # change_controller_type(JP_h5py_file)
        with h5py.File(traj_JP_path, 'r+') as f:
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
        cprint(f"Convertion has been done\n You should find{traj_JP_path}", 'green')

    def traj_JP_eeloss(self, original_path: str):
        '''
        1. copy original_path to traj_JP_path
        2. change controller type to JOINT_POSITION
        3. convert actions from PosAxisOpen to JP
        4. add x0loss group with eePose
        '''
        traj_JP_eeloss_path = original_path.replace('.hdf5', '_traj_JP_eeloss.hdf5')
        cprint(f"Converting\n{original_path}\nto{traj_JP_eeloss_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, traj_JP_eeloss_path)
        self._controller_type_to_JP(traj_JP_eeloss_path)

        # change_controller_type(JP_h5py_file)
        with h5py.File(traj_JP_eeloss_path, 'r+') as f:
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

        # add x0lo
        with h5py.File(traj_JP_eeloss_path, 'r+') as f:
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

        cprint(f"Convertion has been done\n You should find{traj_JP_eeloss_path}", 'green')

    def pure_lowdim_eePose(self, original_path: str):
        pure_lowdim_path = original_path.replace('.hdf5', '_pure_lowdim_traj_eePose.hdf5')
        cprint(f"Converting\n{original_path}\nto{pure_lowdim_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, pure_lowdim_path)
        self._controller_type_to_JP(pure_lowdim_path)
        # change_controller_type(JP_h5py_file)
        with h5py.File(pure_lowdim_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                PosAxisOpen_old = deepcopy(demo_data['actions'][...])  # PosAxis
                robot_ee_pos = deepcopy(demo_data['obs']['robot0_eef_pos'][...])
                robot_ee_quat = deepcopy(demo_data['obs']['robot0_eef_quat'][...])
                isOpen = PosAxisOpen_old[:, -1:]  # open action at t

                # 2. convert to eePose-PosAxis
                PosQuat_curr = np.concatenate([robot_ee_pos, robot_ee_quat], axis=-1)  # eePose at t
                T_obs_curr = PosQuat2HT(PosQuat_curr)
                T_action_curr = T_obs_curr @ inv(PosEuler2HT(self.PosEuler_offset_action2obs[None, ...]))  # offset between action and obs
                PosAxis_curr = HT2PosAxis(T_action_curr)

                # 3. convert to new PosAxisOpen
                PosAxis_new = np.concatenate((PosAxis_curr[1:, :], PosAxisOpen_old[-1:, :-1]), axis=0)
                PosAxisOpen_new = np.concatenate((PosAxis_new, isOpen), axis=-1)

                assert PosAxisOpen_new.shape == PosAxisOpen_old.shape, "PosAxisOpen_new shape is not equal to PosAxisOpen_old shape"

                demo_data['actions'][...] = PosAxisOpen_new
                assert np.all(demo_data['actions'][...] == PosAxisOpen_new), "demo_data['actions'] is not equal to PosAxisOpen_new"

            for i, key in enumerate(data.keys()):
                # delete all obs
                # make states to be the only obs
                demo_data = data[key]
                states = deepcopy(demo_data['states'][...])
                obs_group = demo_data['obs']

                for name, obj in list(obs_group.items()):
                    if isinstance(obj, h5py.Dataset) and name != 'agentview_image':
                        del obs_group[name]

                obs_group.create_dataset('states', data=states, dtype='f8')

    def pure_lowdim_JP(self, original_path: str):
        pure_lowdim_path = original_path.replace('.hdf5', '_pure_lowdim_traj_JP.hdf5')
        cprint(f"Converting\n{original_path}\nto{pure_lowdim_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, pure_lowdim_path)

        # change_controller_type(JP_h5py_file)
        with h5py.File(pure_lowdim_path, 'r+') as f:
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

            for i, key in enumerate(data.keys()):
                # delete all obs
                # make states to be the only obs
                demo_data = data[key]
                states = deepcopy(demo_data['states'][...])
                obs_group = demo_data['obs']

                for name, obj in list(obs_group.items()):
                    if isinstance(obj, h5py.Dataset) and name != 'agentview_image':
                        del obs_group[name]

                obs_group.create_dataset('states', data=states, dtype='f8')

    def JP2eePose(self, original_path: str):
        JP2eePose_path = original_path.replace('.hdf5', '_JP2eePose.hdf5')
        cprint(f"Converting\n{original_path}\nto{JP2eePose_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, JP2eePose_path)
        with h5py.File(JP2eePose_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                JP_curr = deepcopy(demo_data['obs']['robot0_joint_pos'][...])
                pos_ee = deepcopy(demo_data['obs']['robot0_eef_pos'][...])
                quat_ee = deepcopy(demo_data['obs']['robot0_eef_quat'][...])
                open_action = deepcopy(demo_data['actions'][..., -1:])  # open action at t

                # 2. convert to eePose-PosAxis
                PosQuat_curr = np.concatenate([pos_ee, quat_ee], axis=-1)  # eePose at t
                PosAxis_curr = self._PosQuat2PosAxis(PosQuat_curr)

                PosAxis_new = np.concatenate((PosAxis_curr[1:, :], PosAxis_curr[-1:, :]), axis=0)
                PosAxisOpen_new = np.concatenate((PosAxis_new, open_action[:, -1:]), axis=-1)

                JP_new = np.concatenate((JP_curr[1:, :], JP_curr[-1:, :]), axis=0)
                JPOpen_new = np.concatenate((JP_new, open_action), axis=-1)

                demo_data['actions'][...] = PosAxisOpen_new
                for obs_key in list(demo_data['obs'].keys()):
                    del demo_data['obs'][obs_key]

                demo_data['obs'].create_dataset('JPOpen', data=JPOpen_new)
        HDF5Inspector.inspect_hdf5(JP2eePose_path)

    def JP2eePose_debug(self, original_path: str):
        JP2eePose_path = original_path.replace('.hdf5', '_JP2eePose_degbug.hdf5')
        cprint(f"Converting\n{original_path}\nto{JP2eePose_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, JP2eePose_path)
        with h5py.File(JP2eePose_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                length = demo_data['actions'].shape[0]

                arr_eePose = np.arange(length).reshape(-1, 1)  # shape (108, 1)
                arr_eePose = np.tile(arr_eePose, (1, 7))          # shape (108, 7)，每列是行索引

                arr_JPOpen = np.arange(length).reshape(-1, 1)  # shape (108, 1)
                arr_JPOpen = np.tile(arr_JPOpen, (1, 8))          # shape (108, 8)，每列是行索引
                demo_data['actions'][...] = arr_eePose
                for obs_key in list(demo_data['obs'].keys()):
                    del demo_data['obs'][obs_key]
                demo_data['obs'].create_dataset('JPOpen', data=arr_JPOpen)

    ################
    # private method
    ################

    def _PosQuat2PosAxis(self, PosQuat):
        """
        Convert PosQuat to PosAxis
        :param PosQuat: (N, 7) numpy array, where N is the number of samples
        :return: (N, 6) numpy array, where N is the number of samples
        """
        T_obs = PosQuat2HT(PosQuat)
        T_action = T_obs @ inv(PosEuler2HT(self.PosEuler_offset_action2obs[None, ...]))  # offset between action and obs
        PosAxis = HT2PosAxis(T_action)
        return PosAxis

    @staticmethod
    def _copy2new_h5py_file(src_path, dst_path):
        with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
            for name in src:
                src.copy(name, dst, name)
        cprint(f"Copied {src_path} to {dst_path}", 'green')

    def _controller_type_to_JP(self, path: str):
        with h5py.File(path, 'r+') as f:
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


if __name__ == '__main__':
    convertor = DatasetConvertor()
    convertor.traj_eePose('data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5')
    # convertor.traj_JP('data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5')
    # convertor.traj_JP_eeloss('data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5')
    # convertor.pure_lowdim_JP('data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5')
    # convertor.JP2eePose_debug('data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5')
