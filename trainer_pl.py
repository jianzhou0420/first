# helper package
# try:
#     import warnings
#     warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")
#     warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_bwd.*")
#     warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
# except:
#     pass

import pathlib
import gym  # Or your custom environment library
import time
from datetime import datetime
import os
import os
from typing import Type, Dict, Any
import copy

# framework package
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset
# equidiff package
from equi_diffpo.workspace.base_workspace import BaseWorkspace
from equi_diffpo.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from equi_diffpo.dataset.base_dataset import BaseImageDataset
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.common.checkpoint_util import TopKCheckpointManager
from equi_diffpo.common.json_logger import JsonLogger
from equi_diffpo.common.pytorch_util import dict_apply, optimizer_to
from equi_diffpo.model.diffusion.ema_model import EMAModel
from equi_diffpo.model.common.lr_scheduler import get_scheduler
# Hydra specific imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from config_hint import AppConfig

from equi_diffpo.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
import sys
from termcolor import cprint
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['HYDRA_FULL_ERROR'] = "1"

torch.set_float32_matmul_precision('medium')

# ---------------------------------------------------------------
# region 1. Trainer


class Trainer_all(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        policy: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        policy_ema: DiffusionUnetHybridImagePolicy = copy.deepcopy(policy)

        if cfg.training.use_ema:
            ema_handler: EMAModel = hydra.utils.instantiate(
                cfg.ema,
                model=policy_ema,)

        self.policy = policy
        self.policy_ema = policy_ema
        self.ema_handler = ema_handler
        self.train_sampling_batch = None

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.normalizer = self.trainer.datamodule.normalizer
            self.policy.set_normalizer(self.normalizer)
            self.policy_ema.set_normalizer(self.normalizer) if self.cfg.training.use_ema else None

        return

    def training_step(self, batch):
        if self.train_sampling_batch is None:
            self.train_sampling_batch = batch

        loss = self.policy.compute_loss(batch)
        self.logger.experiment.log({
            'train/train_loss': loss.item(),
            'train/lr': self.optimizers().param_groups[0]['lr'],
            'trainer/global_step': self.global_step,
        }, step=self.global_step)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This hook is called after the training step and optimizer update.
        It's the perfect place to update the EMA weights.
        """
        self.ema_handler.step(self.policy)

    def validation_step(self, batch):
        loss = self.policy_ema.compute_loss(batch)
        self.logger.experiment.log({
            'train/val_loss': loss.item(),
        }, step=self.global_step)
        return loss

    def configure_optimizers(self):
        cfg = self.cfg
        num_training_steps = self.trainer.estimated_stepping_batches

        optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.policy.parameters())
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                num_training_steps * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # Make sure to step the scheduler every batch/step
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        This hook is called when a checkpoint is saved.
        We replace the full state_dict with ONLY the state_dict of the policy
        we want to save (either the training policy or the EMA one).
        """
        if self.cfg.training.use_ema:
            # Get the state_dict from your EMA model
            policy_state_to_save = self.policy_ema.state_dict()
        else:
            # Get the state_dict from the standard training model
            policy_state_to_save = self.policy.state_dict()

        # Overwrite the complete state_dict with only the policy's state
        checkpoint['state_dict'] = policy_state_to_save
# endregion
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# region 2. DataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == 'fit':
            cfg = self.cfg
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            val_dataset = dataset.get_validation_dataset()

            assert isinstance(dataset, BaseImageDataset)
            normalizer = dataset.get_normalizer()

            self.normalizer = normalizer
            self.dataset = dataset
            self.val_dataset = val_dataset

    def train_dataloader(self):
        train_dataloader = DataLoader(self.dataset, **self.cfg.dataloader)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, **self.cfg.val_dataloader)
        return val_dataloader


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 3. Callback


class RolloutCallback(pl.Callback):
    """
    A Callback to run a policy rollout in an environment periodically.
    """

    def __init__(self, cfg: AppConfig):
        super().__init__()
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir='data/outputs')  # TODO:fix it
        assert isinstance(env_runner, BaseImageRunner)

        self.rollout_every_n_epochs = cfg.training.rollout_every
        self.env_runner = env_runner

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Trainer_all):
        """
        This hook is called after every validation epoch.
        """

        # Ensure we only run this every N epochs
        if (trainer.current_epoch) + 1 % self.rollout_every_n_epochs != 0:
            return
        runner_log = self.env_runner.run(pl_module.policy_ema)
        trainer.logger.experiment.log(runner_log, step=trainer.global_step)


class ActionMseLossForDiffusion(pl.Callback):
    """
    A Callback to compute the MSE loss of actions in the diffusion model.
    This is useful for training the diffusion model with action data.
    """

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Trainer_all):
        """
        This hook is called after every validation epoch.
        """
        if pl_module.global_step <= 0:
            return
        train_sampling_batch = pl_module.train_sampling_batch

        batch = dict_apply(train_sampling_batch, lambda x: x.to(pl_module.device, non_blocking=True))
        obs_dict = batch['obs']
        gt_action = batch['action']
        result = pl_module.policy_ema.predict_action(obs_dict)
        pred_action = result['action_pred']
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        trainer.logger.experiment.log({
            'train/action_mse_loss': mse,
        }, step=trainer.global_step)


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region Main


def train(cfg: AppConfig):

    # 1. Define a unique name and directory for this specific run
    model_name = 'my_model'  # Or get from cfg
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f"{current_time}_{model_name}"
    output_dir = f"data/outputs/{run_name}"

    # 2. Configure ModelCheckpoint to save in that specific directory
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,  # <-- Tell it exactly where to save
        filename='checkpoint_{epoch:03d}',  # Filename can be simpler now
        every_n_epochs=cfg.training.checkpoint_every,
        save_top_k=-1,
        save_last=False,
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )

    # 3. Configure WandbLogger to use the same directory
    wandb_logger = WandbLogger(
        save_dir=output_dir,  # <-- Use save_dir to point to the same path
        name=run_name,       # <-- Give the run a nice name in the Wandb UI
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging,
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback,
                                    RolloutCallback(cfg),
                                    ActionMseLossForDiffusion(cfg),
                                    ],
                         max_epochs=int(cfg.training.num_epochs),
                         devices='auto',
                         strategy='auto',
                         logger=[wandb_logger],
                         use_distributed_sampler=False,
                         check_val_every_n_epoch=cfg.training.val_every,

                         )
    trainer_model = Trainer_all(cfg)
    data_module = MyDataModule(cfg)
    trainer.fit(trainer_model, datamodule=data_module)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'equi_diffpo', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    train(cfg)


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
    max_steps = {
        'stack_d1': 400,
        'stack_three_d1': 400,
        'square_d2': 400,
        'threading_d2': 400,
        'coffee_d2': 400,
        'three_piece_assembly_d2': 500,
        'hammer_cleanup_d1': 500,
        'mug_cleanup_d1': 500,
        'kitchen_d1': 800,
        'nut_assembly_d0': 500,
        'pick_place_d0': 1000,
        'coffee_preparation_d1': 800,
        'tool_hang': 700,
        'can': 400,
        'lift': 400,
        'square': 400,
    }

    def get_ws_x_center(task_name):
        if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
            return -0.2
        else:
            return 0.

    def get_ws_y_center(task_name):
        return 0.

    OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
    OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)

    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    main()
