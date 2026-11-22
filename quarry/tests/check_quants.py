#!/usr/bin/env python3
# Compare Quarry's dequantized rows with gguf-py's independent dequantizer.
# Run with a Python that has numpy + gguf installed.
import sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

ROWS = 8
r = GGUFReader(sys.argv[1])
by_name = {t.name: t for t in r.tensors}
fails = 0
for name in ["blk.0.attn_q.weight", "blk.5.ffn_down_exps.weight", "output.weight"]:
    t = by_name[name]
    n = int(t.shape[0])
    data = np.asarray(t.data)                 # raw blocks, rows along first axis
    raw = data.reshape(-1, data.shape[-1])[:ROWS]
    ref = dequantize(raw, t.tensor_type).reshape(ROWS, n).astype(np.float32)
    got = np.fromfile(f"tests/out_{name}.f32", dtype=np.float32).reshape(ROWS, n)
    err = np.abs(ref - got).max()
    ok = err <= 1e-6 * max(1.0, np.abs(ref).max())
    print(f"{name:28s} max abs diff {err:.3e}  (|ref| max {np.abs(ref).max():.3f})  {'OK' if ok else 'FAIL'}")
    fails += not ok
sys.exit(fails)
