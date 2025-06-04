
import h5py
import numpy as np
from zero.z_utils.h5_utils import copy2new_h5py_file, HDF5Inspector
import json

h5py_path = "/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_JP.hdf5"

# HDF5Inspector.inspect_hdf5(h5py_path)

with h5py.File(h5py_path, 'r') as f:
    env_args = json.loads(f['data'].attrs['env_args'])

    print("Environment Arguments:")
    for key, value in env_args.items():
        print(f"{key}: {value}")
    pass
