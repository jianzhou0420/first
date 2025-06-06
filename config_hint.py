'''
Just for hightlighting the config schema. will not take effect in the code.
'''
# config_schema.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# A generic dictionary for shape_meta, as its keys are dynamic
ShapeDict = Dict[str, Dict[str, Any]]

# --- Task-specific Schemas ---


@dataclass
class EnvRunnerConfig:
    _target_: str = "equi_diffpo.env_runner.robomimic_image_runner.RobomimicImageRunner"
    dataset_path: str = "???"  # To be defined by interpolation
    shape_meta: ShapeDict = field(default_factory=dict)
    n_train: int = 6
    n_train_vis: int = 2
    train_start_idx: int = 0
    n_test: int = 50
    n_test_vis: int = 4
    test_start_seed: int = 100000
    max_steps: Any = "???"  # Interpolated
    n_obs_steps: int = "???"  # Interpolated
    n_action_steps: int = "???"  # Interpolated
    render_obs_key: str = 'agentview_image'
    fps: int = 10
    crf: int = 22
    past_action: bool = "???"  # Interpolated
    abs_action: bool = True
    tqdm_interval_sec: float = 1.0
    n_envs: int = 28


@dataclass
class TaskDatasetConfig:
    _target_: str = "???"  # Interpolated from main config
    n_demo: int = "???"  # Interpolated
    shape_meta: ShapeDict = field(default_factory=dict)
    dataset_path: str = "???"
    horizon: int = "???"
    pad_before: Any = "???"  # Interpolated
    pad_after: Any = "???"  # Interpolated
    n_obs_steps: int = "???"
    abs_action: bool = True
    rotation_rep: str = 'rotation_6d'
    use_legacy_normalizer: bool = False
    use_cache: int = 0
    seed: int = 42
    val_ratio: float = 0.02


@dataclass
class TaskConfig:
    # This class represents the 'mimicgen_abs.yaml' file
    name: str = "mimicgen_abs"
    shape_meta: ShapeDict = field(default_factory=dict)
    abs_action: bool = True
    env_runner: EnvRunnerConfig = field(default_factory=EnvRunnerConfig)
    dataset: TaskDatasetConfig = field(default_factory=TaskDatasetConfig)


# --- Main Application Schemas ---
@dataclass
class NoiseSchedulerConfig:
    _target_: str = "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"
    num_train_timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    variance_type: str = "fixed_small"
    clip_sample: bool = True
    prediction_type: str = "epsilon"


@dataclass
class PolicyConfig:
    _target_: str = "equi_diffpo.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy"
    shape_meta: ShapeDict = field(default_factory=dict)
    noise_scheduler: NoiseSchedulerConfig = field(default_factory=NoiseSchedulerConfig)
    horizon: Any = "???"  # Interpolated
    n_action_steps: Any = "???"  # Interpolated
    n_obs_steps: Any = "???"  # Interpolated
    num_inference_steps: int = 100
    obs_as_global_cond: Any = "???"  # Interpolated
    crop_shape: Optional[List[int]] = field(default_factory=lambda: [76, 76])
    diffusion_step_embed_dim: int = 128
    down_dims: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    obs_encoder_group_norm: bool = True
    eval_fixed_crop: bool = True
    rot_aug: bool = False


@dataclass
class EMAConfig:
    _target_: str = "equi_diffpo.model.diffusion.ema_model.EMAModel"
    update_after_step: int = 0
    inv_gamma: float = 1.0
    power: float = 0.75
    min_value: float = 0.0
    max_value: float = 0.9999


@dataclass
class DataLoaderConfig:
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 1.0e-4
    betas: List[float] = field(default_factory=lambda: [0.95, 0.999])
    eps: float = 1.0e-8
    weight_decay: float = 1.0e-6


@dataclass
class TrainingConfig:
    device: str = "cuda:0"
    seed: int = 0
    debug: bool = False
    resume: bool = True
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    num_epochs: int = "???"  # Interpolated
    gradient_accumulate_every: int = 1
    use_ema: bool = True
    rollout_every: Any = "???"
    checkpoint_every: Any = "???"
    val_every: int = 1
    sample_every: int = 5
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None
    tqdm_interval_sec: float = 1.0


@dataclass
class LoggingConfig:
    project: Any = "???"  # Interpolated
    resume: bool = True
    mode: str = "online"
    name: Any = "???"  # Interpolated
    tags: List[str] = field(default_factory=list)
    id: Optional[str] = None
    group: Optional[str] = None


@dataclass
class CheckpointTopKConfig:
    monitor_key: str = "test_mean_score"
    mode: str = "max"
    k: int = 5
    format_str: str = 'epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt'


@dataclass
class CheckpointConfig:
    topk: CheckpointTopKConfig = field(default_factory=CheckpointTopKConfig)
    save_last_ckpt: bool = True
    save_last_snapshot: bool = False


@dataclass
class HydraJobConfig:
    override_dirname: Any = "${name}"


@dataclass
class HydraRunConfig:
    dir: str = "data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}"


@dataclass
class HydraSweepConfig:
    dir: str = "data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}"
    subdir: str = "${hydra.job.num}"


@dataclass
class HydraConfig:
    job: HydraJobConfig = field(default_factory=HydraJobConfig)
    run: HydraRunConfig = field(default_factory=HydraRunConfig)
    sweep: HydraSweepConfig = field(default_factory=HydraSweepConfig)


@dataclass
class MultiRunConfig:
    run_dir: str = "data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}"
    wandb_name_base: str = "${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}"


# --- Top-Level Application Config ---
@dataclass
class AppConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        {"task": "mimicgen_abs"}
    ])

    # Primary Configs
    name: str = "diff_c"
    _target_: str = "equi_diffpo.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace"
    shape_meta: ShapeDict = field(default_factory=dict)
    exp_name: str = "default"
    task_name: str = "stack_d1"
    n_demo: int = 200
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    n_latency_steps: int = 0
    dataset_obs_steps: int = "${n_obs_steps}"
    past_action_visible: bool = False
    obs_as_global_cond: bool = True
    dataset: str = "equi_diffpo.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset"
    dataset_path: str = "data/robomimic/datasets/${task_name}/${task_name}_abs.hdf5"

    # Nested Sections
    task: TaskConfig = field(default_factory=TaskConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    val_dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    multi_run: MultiRunConfig = field(default_factory=MultiRunConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)
