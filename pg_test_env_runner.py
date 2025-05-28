
"""
    env_runner:
      _target_: equi_diffpo.env_runner.robomimic_image_runner.RobomimicImageRunner
      dataset_path: data/robomimic/datasets/stack_d1/stack_d1_abs_test.hdf5
      shape_meta:
        obs:
          agentview_image:
            shape:
            - 3
            - 84
            - 84
            type: rgb
          robot0_eye_in_hand_image:
            shape:
            - 3
            - 84
            - 84
            type: rgb
          robot0_eef_pos:
            shape:
            - 3
          robot0_eef_quat:
            shape:
            - 4
          robot0_gripper_qpos:
            shape:
            - 2
        action:
          shape:
          - 10
      n_train: 6
      n_train_vis: 2
      train_start_idx: 0
      n_test: 50
      n_test_vis: 4
      test_start_seed: 100000
      max_steps: 400
      n_obs_steps: 2
      n_action_steps: 8
      render_obs_key: agentview_image
      fps: 10
      crf: 22
      past_action: false
      abs_action: true
      tqdm_interval_sec: 1.0
      n_envs: 28


"""


import time
from equi_diffpo.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from equi_diffpo.env_runner.robomimic_image_runner_JP import RobomimicImageRunner
from equi_diffpo.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

import pickle
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DebugPolicyCreator:
    @staticmethod
    def get_a_trained_policy(path='/media/jian/ssd4t/DP/first/data/outputs/2025.05.21/18.24.54_diff_c_stack_d1/checkpoints/epoch=0250-test_mean_score=0.660.ckpt'):
        """
        Get a trained policy for testing.
        :return: DiffusionUnetHybridImagePolicy
        """
        test = TrainDiffusionUnetHybridWorkspace.create_from_checkpoint(path=path)
        policy = test.model
        policy.eval()

        return policy

    @staticmethod
    def get_a_empty_policy():
        shape_meta = {
            'obs': {
                'agentview_image': {
                    'shape': [3, 84, 84],
                    'type': 'rgb'
                },
                'robot0_eye_in_hand_image': {
                    'shape': [3, 84, 84],
                    'type': 'rgb'
                },
                'robot0_eef_pos': {
                    'shape': [3]
                },
                'robot0_eef_quat': {
                    'shape': [4]
                },
                'robot0_gripper_qpos': {
                    'shape': [2]
                },


            },
            'action': {
                'shape': [8]
            }
        }
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',  # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
            clip_sample=True,  # required when predict_epsilon=False
            prediction_type='epsilon'  # or sample
        )

        # shape_meta: dict,
        # noise_scheduler: DDPMScheduler,
        # horizon,
        # n_action_steps,
        # n_obs_steps,
        # num_inference_steps = None,
        # obs_as_global_cond = True,
        # crop_shape = (76, 76),
        # diffusion_step_embed_dim = 256,
        # down_dims = (256, 512, 1024),
        # kernel_size = 5,
        # n_groups = 8,
        # cond_predict_scale = True,
        # obs_encoder_group_norm = False,
        # eval_fixed_crop = False,
        # rot_aug = False,

        policy = DiffusionUnetHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=100,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=128,
            down_dims=(512, 1024, 2048),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=True,
            eval_fixed_crop=True,
            rot_aug=False
        )
        policy.eval()

        from equi_diffpo.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset

        # configure dataset
        dataset = RobomimicReplayImageDataset(
            n_demo=10,
            shape_meta=shape_meta,
            dataset_path='/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP.hdf5',
            horizon=16,
            pad_before=1,
            pad_after=7,
            n_obs_steps=2,
            abs_action=0,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            use_cache=1,
            seed=42,
            val_ratio=0.02,
        )

        normalizer = dataset.get_normalizer()

        policy.set_normalizer(normalizer)
        return policy, shape_meta

    def get_a_dummy_policy(self):
        from equi_diffpo.policy.base_image_policy import BaseImagePolicy

        class DummyPolicy(BaseImagePolicy):
            pass

        return policy,


if __name__ == '__main__':
    # policy, shape_meta = DebugPolicyCreator.get_a_trained_policy() #TODO
    policy = DebugPolicyCreator.get_a_trained_policy(
        path='/media/jian/ssd4t/DP/first/data/outputs/2025.05.24/23.23.03_diff_c_stack_d1/checkpoints/latest.ckpt'
    ).to('cuda')
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
    dataset_path = "/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP.hdf5"

    env_runner = RobomimicImageRunner(
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        n_train=6,
        n_train_vis=2,
        train_start_idx=0,
        n_test=10,
        n_test_vis=4,
        test_start_seed=100000,
        max_steps=300,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key='agentview_image',
        fps=10,
        crf=22,
        past_action=False,
        abs_action=0,
        tqdm_interval_sec=1.0,
        n_envs=16,
        output_dir='./test'
    )

    log = env_runner.run(policy)

    with open('test.pkl', 'wb') as f:
        pickle.dump(log, f)
