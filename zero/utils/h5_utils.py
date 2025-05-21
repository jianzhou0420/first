import inspect
import sys
import h5py


def copy2new_h5py_file(src_path, dst_path):
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for name in src:
            src.copy(name, dst, name)


class HDF5Inspector:
    '''
    Print the structure of an HDF5 file in a tree format.
    Usage:
    HDF5Inspector.inspect_hdf5('path/to/your/file.hdf5')
    '''
    @staticmethod
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
                HDF5Inspector.print_tree(child_name, child_obj, new_prefix, last)
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{prefix}{connector}{name.split('/')[-1]}  ⟵ Dataset (shape: {shape}, dtype: {dtype})")
            if obj.attrs:
                for i, attr in enumerate(obj.attrs):
                    is_last_attr = (i == len(obj.attrs) - 1)
                    attr_connector = '└── ' if is_last_attr else '├── '
                    print(f"{prefix}{'    ' if is_last else '│   '}{attr_connector}@{attr}  ⟵ Attribute on dataset")

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
