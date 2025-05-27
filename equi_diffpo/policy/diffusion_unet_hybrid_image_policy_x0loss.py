from codebase.z_utils.Rotation_torch import mat2quat
import einops
from zero.FrankaPandaFK_torch import FrankaEmikaPanda_torch
from typing import Dict
import math
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
from zero.z_utils.coding import extract
from copy import deepcopy

from codebase.z_utils.Rotation_torch import PosEuler2HT, HT2eePose
from codebase.z_utils.Rotation_torch import matrix_to_rotation_6d, eePose2HT


class DiffusionUnetHybridImagePolicyX0loss(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 crop_shape=(76, 76),
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 obs_encoder_group_norm=False,
                 eval_fixed_crop=False,
                 rot_aug=False,
                 # parameters passed to step
                 **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
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
        self.rot_randomizer = RotRandomizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

        # X0loss
        self.loss_plugin = X0LossPlugin(scheduler=noise_scheduler)
        # /X0loss

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
            # reshape back to B, To, Do
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

        # x0loss
        eePose_GT = deepcopy(batch['x0loss'])

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
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
                                                eePose_GT_with_open=eePose_GT,
                                                normalizer=self.normalizer
                                                )

        # /X0LossPlugin
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss + x0Loss  # add X0 loss to the total loss


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
        PosEuler_base_mimicgen = torch.tensor([-0.561, 0., 0.925, 0., 0., 0.], device='cuda')
        PosEuler_offset = torch.tensor([0., 0., 0., 0., 0., - 180.], device='cuda')
        T_base_mimicgen = PosEuler2HT(PosEuler_base_mimicgen[None, ...])[0]
        T_offset = PosEuler2HT(PosEuler_offset[None, ...])[0]

        franka = FrankaEmikaPanda_torch()
        franka.set_T_base(T_base_mimicgen)
        franka.set_T_offset(T_offset)
        self.sigmoid = nn.Sigmoid()

        self.franka = franka
        lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

        self.lower_t = torch.tensor(lower, device='cuda')
        self.upper_t = torch.tensor(upper, device='cuda')

    def inverse_q_sample(self, x_t, t, noise):
        '''
        inverse diffusion process, it is not denoising! Just for apply pysical rules
        '''
        sqrt_alphas_bar = extract(self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        x_0 = (x_t - sqrt_one_minus_alphas_bar * noise) / sqrt_alphas_bar
        return x_0

    def eePoseMseLoss(self, x_t, t, noise, eePose_GT_with_open, normalizer: LinearNormalizer):
        '''
        x: JP: [B,H,8]
        eePose:[B,H, 3+x+1], 默认都是normalized的

        '''
        # x_t -> x_0
        x_0_pred = self.inverse_q_sample(x_t, t, noise)

        # denormalize x_0 为JP
        JP_with_open_pred = normalizer.unnormalize({'action': x_0_pred})['action']
        JP_pred = JP_with_open_pred[..., :-1]
        JP_pred = JP_pred.clamp(self.lower_t, self.upper_t)  # clamp to valid range
        isopen_pred = self.sigmoid(JP_with_open_pred[..., -1:])  # TODO：remove sigmoid

        T_pred = self.franka.theta2HT(JP_pred, apply_offset=True)
        pos_pred = T_pred[..., :3, 3]
        mat_pred = T_pred[..., :3, :3]
        orthod6d_pred = matrix_to_rotation_6d(mat_pred)
        PosOrthod6d_pred = torch.cat([pos_pred, orthod6d_pred, isopen_pred], dim=-1)

        isopen_GT = eePose_GT_with_open[..., -1:]
        eePose_GT = eePose_GT_with_open[..., :-1]

        T_GT = eePose2HT(eePose_GT)
        pos_GT = T_GT[..., :3, 3]
        mat_GT = T_GT[..., :3, :3]
        orthod6d_GT = matrix_to_rotation_6d(mat_GT)
        PosOrthod6d_GT = torch.cat([pos_GT, orthod6d_GT, isopen_GT], dim=-1)

        loss = F.mse_loss(PosOrthod6d_pred, PosOrthod6d_GT, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss


if __name__ == "__main__":
    pass
