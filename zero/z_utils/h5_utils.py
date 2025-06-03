import inspect
import sys
import h5py


def copy2new_h5py_file(src_path, dst_path):
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for name in src:
            src.copy(name, dst, name)


class HDF5Inspector:
    '''
    Print the structure of an HDF5 file in a tree format,
    but limit each group to displaying only 10 child keys.
    Usage:
    HDF5Inspector.inspect_hdf5('path/to/your/file.hdf5')
    '''
    MAX_KEYS = 15  # Maximum number of child items to display per group # TODO： make this a parameter

    @staticmethod
    def print_tree(name, obj, prefix='', is_last=True):
        connector = '└── ' if is_last else '├── '
        # Print either the root or a group/dataset name
        if isinstance(obj, h5py.Group):
            if name == '/':
                print(name)
            else:
                print(f"{prefix}{connector}{name.split('/')[-1]}")
            # Print any attributes on the group
            if obj.attrs:
                for i, attr in enumerate(obj.attrs):
                    is_last_attr = (i == len(obj.attrs) - 1 and len(obj) == 0)
                    attr_connector = '└── ' if is_last_attr else '├── '
                    attr_prefix = prefix + ('    ' if is_last else '│   ')
                    print(f"{attr_prefix}{attr_connector}@{attr}  ⟵ Attribute on group")
            # Get child items, but limit to MAX_KEYS
            items = list(obj.items())
            total_children = len(items)
            items_to_print = items[:HDF5Inspector.MAX_KEYS]
            for idx, (child_name, child_obj) in enumerate(items_to_print):
                last = (idx == len(items_to_print) - 1) and (total_children <= HDF5Inspector.MAX_KEYS)
                new_prefix = prefix + ('    ' if is_last else '│   ')
                HDF5Inspector.print_tree(child_name, child_obj, new_prefix, last)
            # If there are more than MAX_KEYS children, indicate truncation
            if total_children > HDF5Inspector.MAX_KEYS:
                trunc_prefix = prefix + ('    ' if is_last else '│   ')
                print(f"{trunc_prefix}└── ... and {total_children - HDF5Inspector.MAX_KEYS} more items")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{prefix}{connector}{name.split('/')[-1]}  ⟵ Dataset (shape: {shape}, dtype: {dtype})")
            # Print any attributes on the dataset
            if obj.attrs:
                for i, attr in enumerate(obj.attrs):
                    is_last_attr = (i == len(obj.attrs) - 1)
                    attr_connector = '└── ' if is_last_attr else '├── '
                    attr_prefix = prefix + ('    ' if is_last else '│   ')
                    print(f"{attr_prefix}{attr_connector}@{attr}  ⟵ Attribute on dataset")

    @staticmethod
    def inspect_hdf5(file_path):
        """Inspect and print the structure of the given HDF5 file."""
        try:
            with h5py.File(file_path, 'r') as f:
                HDF5Inspector.print_tree('/', f['/'], '', True)
        except Exception as e:
            print(f"Error opening file '{file_path}': {e}")


def list_current_defs():
    mod = sys.modules[__name__]

    # 1) Top-level functions
    print("Functions:")
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        # only functions defined in this file
        if fn.__module__ == mod.__name__:
            print(f"  {name}")

    # 2) Classes + their methods
    print("\nClasses & methods:")
    for cname, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ == mod.__name__:
            print(f"  Class: {cname}")
            for mname, meth in inspect.getmembers(cls, inspect.isfunction):
                # only methods defined on this class
                if meth.__qualname__.startswith(cname + "."):
                    print(f"    {mname}")
