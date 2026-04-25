# explore_dataset.py
import h5py
import numpy as np

def explore_h5(path):
    with h5py.File(path, 'r') as f:
        print("=" * 60)
        print(f"TOP-LEVEL KEYS: {list(f.keys())}")
        print("=" * 60)

        def visitor(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}[DATASET] {name}")
                print(f"{indent}          shape={obj.shape}, dtype={obj.dtype}")
                # peek at a slice
                try:
                    sample = obj[0] if obj.ndim >= 1 else obj[()]
                    print(f"{indent}          sample[0]={sample}")
                except:
                    pass
            elif isinstance(obj, h5py.Group):
                print(f"{indent}[GROUP]   {name}  ({len(obj)} children)")

        f.visititems(visitor)

        print("\n" + "=" * 60)
        print("ATTRIBUTE SCAN (top-level)")
        with h5py.File(path, 'r') as ff:
            for k, v in ff.attrs.items():
                print(f"  {k}: {v}")

explore_h5('data/airbus_heli/dftrain.h5')
