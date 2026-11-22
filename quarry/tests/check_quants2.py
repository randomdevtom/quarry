# compare build/deq_*.f32 with gguf-py dequantization (run with vendor gguf-py on the path)
import sys, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(sys.argv[1]); T = {t.name: t for t in r.tensors}
for line in open(sys.argv[2]):
    path, name, _ = line.split()
    t = T[name]; n = int(t.shape[0]); got = np.fromfile(path, np.float32).reshape(8, n)
    raw = np.asarray(t.data).reshape(-1, t.data.shape[-1]) if t.data.ndim > 1 else t.data
    ref = dequantize(np.asarray(t.data)[:8] if t.data.ndim == 2 else np.asarray(t.data).reshape(-1, t.data.shape[-1])[:8], t.tensor_type).reshape(8, n)
    print(f"{name:34s} {t.tensor_type.name:5s} max |diff| {np.abs(got-ref).max():.3g}  {'OK' if np.array_equal(got, ref) else 'DIFF'}")
