import einops
from zero.expForwardKinematics.ReconLoss.FrankaPandaFK_torch import FrankaEmikaPanda_torch
from zero.z_utils.coding import extract
from zero.expForwardKinematics.models.Base.BaseAll import BasePolicy
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import equi_diffpo.model.vision.crop_randomizer as dmvc
from equi_diffpo.common.pytorch_util import dict_apply, replace_submodules
from equi_diffpo.model.vision.rot_randomizer import RotRandomizer


from equi_diffpo.policy.base_image_policy import BaseImagePolicy


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: MultiImageObsEncoder,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 # parameters passed to step
                 **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # X0LossPlugin
        self.loss_plugin = X0LossPlugin(scheduler=noise_scheduler)
    # ========= inference  ============

    def conditional_sample(self,
                           condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t,
                                 local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        obs_dict = obs_dict['obs']
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # X0LossPlugin
        x0Loss = self.loss_plugin.eePoseMseLoss(x_t=noisy_trajectory,
                                                t=timesteps,
                                                noise=pred,
                                                eePose=batch['eePose'])

        # /X0LossPlugin

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        total_loss = loss + x0Loss
        return total_loss


class X0LossPlugin(nn.Module):  # No trainable parameters, inherit from nn.Module just for coding

    def __init__(self, scheduler: DDPMScheduler):
        super().__init__()
        self.scheduler = scheduler

        # diffusion model
        def rb(name, val): return self.register_buffer(name, val)  # 这一步太天才了
        max_t = len(scheduler.timesteps)
        betas = scheduler.betas
        alphas = scheduler.alphas
        alphas_bar = scheduler.alphas_cumprod
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:max_t]

        gripper_loc_bounds = torch.tensor(
            [[-0.11010312, -0.5557904, 0.71287104],
             [0.64813266, 0.51835946, 1.51160351]], device='cuda')

        rb('gripper_loc_bounds', gripper_loc_bounds)
        rb('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        rb('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # denoising coeffs
        rb('coeff1', torch.sqrt(1. / alphas))
        rb('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # D-H Parameters
        self.franka = FrankaEmikaPanda_torch()
        lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
        upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1]

        self.lower_t = torch.tensor(lower, device='cuda')
        self.upper_t = torch.tensor(upper, device='cuda')

        self.sigmoid = nn.Sigmoid()

    def inverse_q_sample(self, x_t, t, noise):
        '''
        inverse diffusion process, it is not denoising! Just for apply pysical rules
        '''
        sqrt_alphas_bar = extract(self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        x_0 = (x_t - sqrt_one_minus_alphas_bar * noise) / sqrt_alphas_bar
        return x_0

    def eePoseMseLoss(self, x_t, t, noise, eePose, rot_type='ortho6d'):
        '''
        x: JP: [B,H,8]
        eePose:[B,H, 3+x+1], 默认都是normalized的
        '''

        x_0_pred = self.inverse_q_sample(x_t, t, noise)
        JP_with_open_pred = self.denormalize_JP(x_0_pred)
        JP_pred = JP_with_open_pred[..., :-1]
        isopen_pred = self.sigmoid(JP_with_open_pred[..., -1:])  # TODO：remove sigmoid
        eeHT_pred = self.franka.theta2HT(JP_pred)

        pos_pred = eeHT_pred[..., :3, 3]
        pos_pred = self.normalize_pos(pos_pred)
        rot_pred = eeHT_pred[..., :3, :3]
        if rot_type == 'ortho6d':
            B, H, r, c = rot_pred.shape
            rot_pred = einops.rearrange(rot_pred, 'B H r c -> (B H) r c')
            rot_pred = Ortho6D_torch.get_ortho6d_from_rotation_matrix(rot_pred)
            rot_pred = einops.rearrange(rot_pred, '(B H) r -> B H r', B=B, H=H)
        else:
            NotImplementedError()

        eePose_pred = torch.cat([pos_pred, rot_pred, isopen_pred], dim=-1)

        loss = F.mse_loss(eePose_pred, eePose, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def denormalize_JP(self, norm_JP):
        JP = self.lower_t + (norm_JP + 1) / 2 * (self.upper_t - self.lower_t)
        return JP

    def denormalize_eePose(self, norm_eePose):
        eePose = self.lower_t + (norm_eePose + 1) / 2 * (self.upper_t - self.lower_t)
        return eePose

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0]
        pos_max = self.gripper_loc_bounds[1]
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def denormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0]
        pos_max = self.gripper_loc_bounds[1]
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


class DPWithLossWrapper(BasePolicy):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            if self.config['DP']['ActionHead']['rot_norm_type'] == 'ortho6d':
                len_rot = 6
            elif self.config['DP']['ActionHead']['rot_norm_type'] == 'euler':
                len_rot = 3
            elif self.config['DP']['ActionHead']['rot_norm_type'] == 'quat':
                len_rot = 4

            len_act = 3 + len_rot + 1
            shape_meta = {
                # acceptable types: rgb, low_dim
                "obs": {
                    "image0": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "image1": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "image2": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "image3": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "eePos": {
                        "shape": [3],
                    },
                    "eeRot": {
                        "shape": [len_rot],
                    },
                    "eeOpen": {
                        "shape": [1],
                    },
                },
                "action": {
                    "shape": [len_act],
                },
            }
            normalizer = LinearNormalizer()
            normalizer['image0'] = get_image_range_normalizer()
            normalizer['image1'] = get_image_range_normalizer()
            normalizer['image2'] = get_image_range_normalizer()
            normalizer['image3'] = get_image_range_normalizer()
            normalizer['eePos'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['eeRot'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['eeOpen'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
        elif self.config['DP']['ActionHead']['action_mode'] == 'JP':
            if self.config['DP']['ActionHead']['rot_norm_type'] == 'ortho6d':
                len_rot = 6
            elif self.config['DP']['ActionHead']['rot_norm_type'] == 'euler':
                len_rot = 3
            elif self.config['DP']['ActionHead']['rot_norm_type'] == 'quat':
                len_rot = 4

            len_act = 3 + len_rot + 1
            shape_meta = {
                # acceptable types: rgb, low_dim
                "obs": {
                    "image0": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "image1": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "image2": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "image3": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    },
                    "eePos": {
                        "shape": [3],
                    },
                    "eeRot": {
                        "shape": [len_rot],
                    },
                    "eeOpen": {
                        "shape": [1],
                    },
                    "JP_hist": {
                        "shape": [8],
                    },

                },
                "action": {
                    "shape": [8],
                },
            }
            normalizer = LinearNormalizer()
            normalizer['image0'] = get_image_range_normalizer()
            normalizer['image1'] = get_image_range_normalizer()
            normalizer['image2'] = get_image_range_normalizer()
            normalizer['image3'] = get_image_range_normalizer()
            normalizer['eePos'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['eeRot'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['eeOpen'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['JP_hist'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()

        noise_scheduler = DDPMScheduler(num_train_timesteps=100,
                                        beta_start=0.0001,
                                        beta_end=0.02,
                                        beta_schedule='squaredcos_cap_v2',
                                        variance_type='fixed_small',
                                        clip_sample=True,
                                        prediction_type='epsilon',)
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=get_resnet(name='resnet18', weights=None),
            resize_shape=None,
            crop_shape=None,
            random_crop=False,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )

        DP = DiffusionUnetImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=encoder,
            horizon=self.config['DP']['ActionHead']['horizon'],
            n_action_steps=self.config['DP']['ActionHead']['horizon'],
            n_obs_steps=self.config['DP']['ActionHead']['horizon'],
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )

        # normalizer.fit(batch)

        DP.set_normalizer(normalizer)
        self.DP = DP

    def forward(self, batch, **kwargs):
        return self.DP.compute_loss(batch)

    def inference_one_sample(self, batch):
        results = self.DP.predict_action(batch)

        return results


if __name__ == '__main__':
    def test():
        shape_meta = {
            # acceptable types: rgb, low_dim
            "obs": {
                "image0": {
                    "shape": [3, 256, 256],
                    "type": "rgb",
                },
                "image1": {
                    "shape": [3, 256, 256],
                    "type": "rgb",
                },
                "eePos": {
                    "shape": [3],
                    # type default: low_dim
                },
                "eeRot": {
                    "shape": [4],
                },
                "eeOpen": {
                    "shape": [1],
                },
            },
            "action": {
                "shape": [8],
            },
        }

        noise_scheduler = DDPMScheduler(num_train_timesteps=100,
                                        beta_start=0.0001,
                                        beta_end=0.02,
                                        beta_schedule='squaredcos_cap_v2',
                                        variance_type='fixed_small',
                                        clip_sample=True,
                                        prediction_type='epsilon',)
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=get_resnet(name='resnet18', weights=None),
            resize_shape=None,
            crop_shape=None,
            random_crop=False,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )

        DP = DiffusionUnetImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=encoder,
            horizon=1,
            n_action_steps=8,
            n_obs_steps=2,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )

        normalizer = LinearNormalizer()
        normalizer['image0'] = get_image_range_normalizer()
        normalizer['image1'] = get_image_range_normalizer()

        batch = {
            'obs': {
                'image0': torch.randn(2, 8, 3, 256, 256),
                'image1': torch.randn(2, 8, 3, 256, 256),
                'eePos': torch.randn(2, 8, 3),
                'eeRot': torch.randn(2, 8, 4),
                'eeOpen': torch.randn(2, 8, 1),
            },
            'action': torch.randn(2, 8, 8)
        }
        # normalizer.fit(batch)

        normalizer['eePos'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['eeRot'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['eeOpen'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['action'] = SingleFieldLinearNormalizer.create_identity()

        DP.set_normalizer(normalizer)
        loss = DP.compute_loss(batch)
        print(loss)

    # test()
