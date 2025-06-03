
import h5py
import numpy as np
from zero.z_utils.h5_utils import copy2new_h5py_file, HDF5Inspector
import json

h5py_path = "/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_pure_lowdim_traj_eePose.hdf5"

HDF5Inspector.inspect_hdf5(h5py_path)

with h5py.File(h5py_path, 'r') as f:
    obs = f['data/demo_0/obs']
    test = json.loads(f["data"].attrs["env_args"])
    print(test)
