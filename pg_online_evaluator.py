import os
from tqdm import tqdm
import time
import pathlib
from termcolor import cprint
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
import dill
import collections
import numpy as np
from copy import deepcopy
import h5py
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from zero.config.default import get_config
import json
import mimicgen
from robosuite.controllers import ALL_CONTROLLERS
from copy import deepcopy, copy
from codebase.z_utils.Rotation import *
from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
import wandb.sdk.data_types.video as wv
np.set_printoptions(precision=4, suppress=True)


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def check_and_make(path):
    if not os.path.exists(path):
        os.makedirs(path)


def direct_test():
    options = {}
    eval_config = get_config('/media/jian/ssd4t/DP/first/zero/config/online_eval.yaml')
    print(eval_config)

    with open("mimicgen_env.json", "r") as f:
        all_env = json.load(f)
    with open("mimicgen_robots.json", "r") as f:
        all_robots = json.load(f)

    env_name = eval_config['env_name']
    robot = eval_config['robot']
    has_renderer = eval_config['has_renderer']
    has_offscreen_renderer = eval_config['has_offscreen_renderer']

    options['env_name'] = env_name
    options["robots"] = robot
    options['controller_configs'] = load_controller_config(default_controller=eval_config['controller'])
    options['controller_configs']['control_delta'] = False
    print("options", options)

    env = suite.make(
        **options,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    obs = env.reset()
    env.viewer.set_camera(camera_id=0)
    low, high = env.action_spec
    # do visualization
    action_set = None

    for i in range(10000):
        PosEuler_offset_action2obs = np.array([0, 0, 0, 0, 0, 90])
        ee_pos = obs['robot0_eef_pos']
        ee_rot = obs['robot0_eef_quat']
        joint_pos_cos = obs['robot0_joint_pos_cos']
        joint_pos_sin = obs['robot0_joint_pos_sin']
        JP = np.arctan2(joint_pos_sin, joint_pos_cos)

        print("JP", JP)

        ee_open = [1]

        JP = np.concatenate((JP, ee_open), axis=-1)

        obs, reward, done, _ = env.step(JP)

        env.render()


def validate_dataset(dataset_path):
    # 1. load actions

    # 2, env meta & shape meta
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)
    # disable object state observation
    env_meta['env_kwargs']['use_object_obs'] = False

    if 'abs' in dataset_path:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    print("env_meta", env_meta)

    if 'JP' in dataset_path:
        env_meta['env_kwargs']['controller_configs']['kp'] = 150
        shape_meta = {
            "obs": {
                "agentview_image": {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                },
                "robot0_eye_in_hand_image": {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                },
                "robot0_eef_pos": {
                    "shape": [3],
                    "type": "low_dim",
                },
                "robot0_eef_quat": {
                    "shape": [4],
                    "type": "low_dim",
                },
                "robot0_gripper_qpos": {
                    "shape": [2],
                    "type": "low_dim",
                },
                "robot0_joint_pos": {
                    "shape": [7],
                    "type": "low_dim",
                }

            },
            "action": {
                "shape": [8],
            },
        }
    else:
        shape_meta = {
            "obs": {
                "agentview_image": {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                },
                "robot0_eye_in_hand_image": {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                },
                "robot0_eef_pos": {
                    "shape": [3],
                    "type": "low_dim",
                },
                "robot0_eef_quat": {
                    "shape": [4],
                    "type": "low_dim",
                },
                "robot0_gripper_qpos": {
                    "shape": [2],
                    "type": "low_dim",
                },

            },
            "action": {
                "shape": [7],
            },
        }

    # 3. other params
    output_dir = 'data/debug/validate_dataset'
    render_obs_key = 'agentview_image'
    fps = 10
    crf = 22
    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)
    n_obs_steps = 2
    n_action_steps = 8
    max_steps = 400
    n_envs = 2

    n_train = 6
    train_start_idx = 0
    n_train_vis = 2

    check_and_make(output_dir)
    # 4. create env

    def env_fn():
        robomimic_env = create_env(
            env_meta=env_meta,
            shape_meta=shape_meta
        )
        # Robosuite's hard reset causes excessive memory consumption.
        # Disabled to run more envs.
        # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
        robomimic_env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

    test_env = env_fn()
    cprint(type(test_env), 'green', attrs=['bold'])
    cprint(type(test_env.env), 'green', attrs=['bold'])
    cprint(type(test_env.env.env), 'green', attrs=['bold'])
    test_env.reset()

    with h5py.File(dataset_path, 'r') as f:
        train_idx = train_start_idx
        enable_render = train_idx < n_train_vis
        init_state = f[f'data/demo_{train_idx}/states'][0]
        actions = deepcopy(f[f'data/demo_{train_idx}/actions'][...])
        test_env.env.video_recoder.stop()
        test_env.env.file_path = None
        if enable_render:
            filename = pathlib.Path(output_dir).joinpath(
                'media', wv.util.generate_id() + ".mp4")
            filename.parent.mkdir(parents=False, exist_ok=True)
            filename = str(filename)
            test_env.env.file_path = filename
        test_env.env.env.init_state = init_state
    test_env.reset()

    for i in tqdm(range(len(actions) // n_action_steps + 1)):
        if i * n_action_steps >= len(actions):
            break
        action = actions[i * (n_action_steps):i * (n_action_steps) + n_action_steps]
        obs, reward, done, info = test_env.step(action)

    # for i in tqdm(range(len(actions) // 1 + 1)):
    #     action = actions[i:i + 1]
    #     obs, reward, done, info = test_env.step(action)

    # test_env.reset()
    # test_env.close()
    # time.sleep(4)
    return test_env
    # env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)


if __name__ == "__main__":
    a = validate_dataset('data/robomimic/datasets/stack_d1/stack_d1_abs_JP.hdf5')
    a.close()
