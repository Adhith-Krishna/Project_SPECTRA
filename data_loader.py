# Team 337 - Trailblazers (Airbus Fly Your Ideas 2026)
# The aim of data_loader is train the model based on the clean
# data from Airbus Helicopter Accelerometer Dataset collected by
# Airbus SAS and provided to ETH Zurich.

# ----------------------------------------------------------------------
#  Information
# ----------------------------------------------------------------------
# Stack: JAX,
# Target: GPU and TPU on Google Colab

# ----------------------------------------------------------------------
#  Acknowledgements
# ----------------------------------------------------------------------

# We would like to thank Dr. Olga Fink at EFPL (previously at ETH Zurich),
# for her guidance and assistance with the dataset and for clarifying all
# doubts we had pertaining to it. Her help was critical in helping us choose
# the right dataset as our initial plan to use the C-MAPSS dataset was flawed.
#
# We would also like to thank Airbus SAS for collecting the accelerometer data
# initially. It is exceptionally hard to find credible and trustworthy aerospace
# oriented datasets as most data collected is restricted by export control and
# companies understandly prefer to keep it proprietary. # Having access to a 
# real-world dataset of this quality allowed us to focus on solving the 
# engineering problem itself rather than spending significant effort validating 
# whether the data could be trusted in the first place. That foundation made 
# this work possible.
#
# Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
#
# Based on:
# Airbus SAS. (2020). Airbus Helicopter Accelerometer Dataset. ETH Zurich. 
# https://doi.org/10.3929/ETHZ-B-000415151
# Garcia, G. R., Michau, G., Ducoffe, M., Gupta, J. S., & Fink, O. (2020). 
# Temporal signals to images: Monitoring the condition of industrial assets with 
# deep learning image processing algorithms. arXiv. 
# https://doi.org/10.48550/ARXIV.2005.07031

# ----------------------------------------------------------------------
#  Environment for Google Colab
# ----------------------------------------------------------------------
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import h5py
import numpy as np
import jax 
import jax.numpy as jnp
from jax import jit, random
import jax.lax as lax
from functools import partial
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Iterator, NamedTuple, Tuple

# ----------------------------------------------------------------------
#  Datset Structure
# ----------------------------------------------------------------------
H5_GROUP        = "dftrain"
H5_VALUES_KEY   = "block0_values"
SAMPLING_HZ     = 1024

# ----------------------------------------------------------------------
#  Batch Container
# ----------------------------------------------------------------------
class Batch(NamedTuple):
    x: jax.Array #(B, window_size, C)
    y: jax.Array #(B, )

# ----------------------------------------------------------------------
#  I. Raw Loading to CPU
# ----------------------------------------------------------------------
def _load_raw(h5_path: str) -> np.array:
    with h5py.File(h5_path, "r") as f:
        raw = f[f"{H5_GROUP}/{H5_VALUES_KEY}"][()]
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D (C, T), got {raw.shape}")
    
    data = raw.T if raw.shape[0] < raw.shape [1] else raw
    data = data.astype(np.float32)

    T,C = data.shape
    print(f"[loader] Raw Data: {data.shape} - "
          f"{T/SAMPLING_HZ:.1f}s @ {SAMPLING_HZ}, {C} channels")
    
    return data

# ----------------------------------------------------------------------
#  II. Scaler as JAX Native Parameters
# ----------------------------------------------------------------------
def _fit_scaler(
        data: np.ndarray,
        save_path: str,
) -> Tuple[jax.Array,jax.Array]:
    
    sc = StandardScaler().fit(data)
    joblib.dump(sc, save_path)
    print (f"[loader] Scalar saved -> {save_path}")

    mean = jax.device_put(sc.mean_.astype(np.float32))
    scale = jax.device_put(sc.scale_.astype(np.float32))
    return mean, scale

# ----------------------------------------------------------------------
#  III. Device Placement
# ----------------------------------------------------------------------
def _to_device(data: np.ndarray) -> jax.Array:
    arr = jax.device_put(jnp.asarray(data))
    print(f"[loader] Placed on {arr.devices()} - {arr.nbytes/1e6:.1f} MB")
    return arr

# ----------------------------------------------------------------------
#  IV. JIT Compiled Window with Normalization
# ----------------------------------------------------------------------
@partial(jit, static_argnames=("window_size",))
def _extract_window(
    data:       jax.Array,
    start:      jax.Array,
    mean:       jax.Array,
    scale:      jax.Array,
    window_size:int,
) -> jax.Array:
    
    window = lax.dynamic_slice(
        data,
        start_indices=(start, 0),
        slice_sizes=(window_size, data.shape[1]),
    )
    return (window - mean) / scale   
@partial(jit, static_argnames=("window_size",))
def _extract_batch(
    data:        jax.Array,   # (T, C)
    starts:      jax.Array,   # (B,) int32 — start indices for this batch
    mean:        jax.Array,   # (C,)
    scale:       jax.Array,   # (C,)
    window_size: int,
) -> jax.Array:
    extract_one = partial(_extract_window, data,
                          mean = mean, scale = scale, window_size = window_size)
    return jax.vmap(extract_one)(starts)

# ----------------------------------------------------------------------
#  V. Iterator
# ----------------------------------------------------------------------
def _prefetch(iterator: Iterator[Batch], buffer_size: int=2) -> Iterator[Batch]:
    from collections import deque
    queue: deque = deque()

    def enqueue(n: int):
        for _ in range(n):
            try:
                queue.append(next(Iterator))
            except StopIteration:
                pass

    enqueue(buffer_size)
    while queue:
        batch = queue.popleft()
        enqueue(1)
        yield batch

# ----------------------------------------------------------------------
#  VI. JAXDataLoader
# ----------------------------------------------------------------------
class JAXDataLoader:
    def __init__(
        self,
        data:       jax.Array,
        mean:       jax.Array,
        scale:      jax.Array,
        window_size:int = 1024,
        stride:     int = 128,
        batch_size: int = 64,
        shuffle:    bool = False,
        drop_last:  bool = True,
        seed:       int = 42,
        prefetch:   int = 2,
    ):
        self.data           = data
        self.mean           = mean
        self.scale          = scale 
        self.window_size    = window_size
        self.batch_size     = batch_size
        self.shuffle        = shuffle
        self.drop_last      = drop_last
        self.prefetch_n     = prefetch 
        self._rng_key       = random.PRNGKey(seed)

        T = data.shape[0]
        starts = jnp.arange(0, T - window_size + 1, stride, dtype = jnp.int32)
        self._starts    = starts
        self._n_wins    = len(starts)
        self._labels    = jnp.zeros(batch_size, dtype=jnp.int32)

    def __len__(self) -> int:
        if self.drop_last:
            return self._n_wins // self.batch_size
        return (self._n_wins + self.batch_size + 1) // self.batch_size
    
    def _iter_core(self) -> Iterator[Batch]:
        starts = self._starts
 
        if self.shuffle:
            self._rng_key, subkey = random.split(self._rng_key)
            # on-device permutation — no host round-trip
            perm   = random.permutation(subkey, self._n_wins)
            starts = starts[perm]
 
        bs = self.batch_size
        for i in range(0, self._n_wins - (bs - 1), bs):
            if self.drop_last and i + bs > self._n_wins:
                break
            batch_starts = lax.dynamic_slice(starts, (i,), (bs,))
            x = _extract_batch(
                self.data, batch_starts,
                self.mean, self.scale,
                self.window_size,
            )
            yield Batch(x=x, y=self._labels)
 
    def __iter__(self) -> Iterator[Batch]:
        it = self._iter_core()
        if self.prefetch_n > 0:
            return _prefetch(it, self.prefetch_n)
        return it

# ----------------------------------------------------------------------
#  VII. Multidevice Split
# ----------------------------------------------------------------------
def shard_batch(batch: Batch) -> Batch:
    n = jax.device_count()
    def _shard(arr: jax.Array) -> jax.Array:
        return arr.reshape((n, arr.shape[0]//n) + arr.shape[1:])
    return Batch(x=_shard(batch.x),y=_shard(batch.y))

# ----------------------------------------------------------------------
#  VIII. Public API
# ----------------------------------------------------------------------
def build_dataloaders(
        h5_path:        str,
        window_size:    int = 1024,
        stride:         int = 128,
        val_fraction:   float = 0.15,
        batch_size:     int = 64,
        scaler_save_path: str = "scaler_clean.pkl",
        seed:           int = 42,
        prefetch:       int = 2,
) -> Tuple ["JAXDataLoader", "JAXDataLoader", Tuple[jax.Array, jax.Array]]:
    
    data    = _load_raw(h5_path)
    split   = int(len(data) * (1 - val_fraction))

    mean, scale = _fit_scaler(data[:split], scaler_save_path)

    train_dev   = _to_device(data[:split])
    val_dev     = _to_device(data[split:])

    T_train = train_dev.shape[0]
    T_val   = val_dev.shape[0]
    n_train = (T_train - window_size) // stride + 1
    n_val   = (T_val - window_size) // stride + 1
    print (f"[loader] Train: {n_train:,} windows |  Val: {n_val:,} windows"
           f"(window={window_size}, stride={stride})")
    
    train_loader = JAXDataLoader(
        train_dev, mean, scale,
        window_size = window_size, stride = stride,
        batch_size = batch_size, shuffle = True,
        drop_last=True, seed = seed, prefetch = prefetch,
    )
    val_loader = JAXDataLoader (
        window_size = window_size, stride = stride,
        batch_size = batch_size, shuffle = False,
        drop_last=False, seed = seed, prefetch = prefetch,        
    )
    return train_loader, val_loader, (mean, scale)

# ----------------------------------------------------------------------
#  Test Case
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys, time

    h5 = sys.argv[1] if len(sys.argv) > 1 else "dftrain.h5"
 
    print(f"\nJAX backend : {jax.default_backend()}")
    print(f"Devices     : {jax.devices()}\n")
 
    train_loader, val_loader, (mean, scale) = build_dataloaders(
        h5, batch_size=64,
    )

    print("[loader] Warming up JIT...")
    batch = next(iter(train_loader))
    batch.x.block_until_ready()
    print(f"[loader] JIT warm-up done")
 
    t0 = time.perf_counter()
    n  = 0
    for batch in train_loader:
        batch.x.block_until_ready()
        n += 1
    elapsed = time.perf_counter() - t0
 
    print(f"\nBatch x : {batch.x.shape}  dtype={batch.x.dtype}")
    print(f"Batch y : {batch.y.shape}  dtype={batch.y.dtype}")
    print(f"mean={float(batch.x.mean()):.4f}  std={float(batch.x.std()):.4f}")
    print(f"\nThroughput: {n / elapsed:.1f} batches/s  "
          f"({n * 64 * 1024 / elapsed / 1e6:.1f}M samples/s)")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print("\n-- data_loader.py OK --")