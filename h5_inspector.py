import h5py
import argparse


if __name__ == '__main__':
    default_path = '/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_voxel_abs_test.hdf5'  # change this to your default file
    HDF5Inspector.inspect_hdf5(default_path)
