import h5py
import argparse


def print_tree(name, obj, prefix='', is_last=True):
    connector = '└── ' if is_last else '├── '
    if isinstance(obj, h5py.Group):
        if name == '/':
            print(name)
        else:
            print(f"{prefix}{connector}{name.split('/')[-1]}")
        if obj.attrs:
            for i, attr in enumerate(obj.attrs):
                is_last_attr = (i == len(obj.attrs) - 1 and len(obj) == 0)
                attr_connector = '└── ' if is_last_attr else '├── '
                print(f"{prefix}{'    ' if is_last else '│   '}{attr_connector}@{attr}  ⟵ Attribute on group")
        items = list(obj.items())
        for idx, (child_name, child_obj) in enumerate(items):
            last = (idx == len(items) - 1)
            new_prefix = prefix + ('    ' if is_last else '│   ')
            print_tree(child_name, child_obj, new_prefix, last)
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        print(f"{prefix}{connector}{name.split('/')[-1]}  ⟵ Dataset (shape: {shape}, dtype: {dtype})")
        if obj.attrs:
            for i, attr in enumerate(obj.attrs):
                is_last_attr = (i == len(obj.attrs) - 1)
                attr_connector = '└── ' if is_last_attr else '├── '
                print(f"{prefix}{'    ' if is_last else '│   '}{attr_connector}@{attr}  ⟵ Attribute on dataset")


def inspect_hdf5(file_path):
    """Inspect and print the structure of the given HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            print_tree('/', f['/'], '', True)
    except Exception as e:
        print(f"Error opening file '{file_path}': {e}")


def main(default_path=None):
    parser = argparse.ArgumentParser(description='Inspect HDF5 file structure')
    parser.add_argument('file', nargs='?', default=default_path,
                        help='Path to HDF5 file (overrides default)')
    args = parser.parse_args()
    inspect_hdf5(args.file)


if __name__ == '__main__':
    default_path = '/media/jian/ssd4t/DP/equidiff/data/robomimic/datasets/stack_d1/stack_d1_voxel_abs.hdf5'  # change this to your default file
    main(default_path=default_path)
