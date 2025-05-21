import os
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from zero.config.default import get_config
import json
import mimicgen
from robosuite.controllers import ALL_CONTROLLERS
from copy import deepcopy, copy
from codebase.z_utils.Rotation import *


np.set_printoptions(precision=4, suppress=True)
if __name__ == "__main__":
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
