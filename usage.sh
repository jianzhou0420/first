# Template
dataset="coffee_d2"

#obs
# python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/${dataset}/${dataset}.hdf5 --output data/robomimic/datasets/${dataset}/${dataset}_voxel.hdf5 --num_workers=18

#action
# python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_abs.hdf5 -n 18
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/${dataset}/${dataset}_voxel.hdf5 -o data/robomimic/datasets/${dataset}/${dataset}_voxel_abs.hdf5 -n 18

# train

python train.py --config-name=train_diffusion_unet_JP task_name=stack_d1 n_demo=100

# debug args
logging.mode=offline task.env_runner.n_train=1 task.env_runner.n_test=1 task.env_runner.n_envs=2
