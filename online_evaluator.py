import os
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from zero.config.default import get_config
import json
import mimicgen_envs
from robosuite.controllers import ALL_CONTROLLERS
from copy import deepcopy, copy
from codebase.z_utils.Rotation import euler2quat, quat2euler, quat2axisangle


np.set_printoptions(precision=4, suppress=True)
if __name__ == "__main__":
    options = {}
    eval_config = get_config('/media/jian/ssd4t/DP/equidiff/zero/config/online_eval.yaml')
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

    env.reset()
    env.viewer.set_camera(camera_id=0)
    low, high = env.action_spec
    # do visualization
    action_set = None

    angle = np.radians(np.array([0, 0, -90]))
    quat = quat2axisangle(euler2quat(angle))
    action = np.array([0.1, 0.1, 1.3, quat[0], quat[1], quat[2], 1])
    # action = np.array([0.1, 0.1, 1.3, 1, 0, 0, 40])
    print("target_action", action)
    for i in range(10000):
        obs, reward, done, _ = env.step(action)
        ee_pos = obs['robot0_eef_pos']
        ee_rot = obs['robot0_eef_quat']
        ee_rot = np.array([ee_rot[0], ee_rot[1], ee_rot[2], ee_rot[3]])
        ee_rot = np.degrees(quat2euler(ee_rot))
        current_eePose = np.concatenate((ee_pos, ee_rot))

        print(action, current_eePose)
        env.render()
