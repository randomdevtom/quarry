# compare Quarry debug dumps (prefix_q) with llama.cpp dumps (prefix_r) for one slot
import sys, numpy as np, os, glob
q, r, slot = sys.argv[1], sys.argv[2], sys.argv[3]
for name in ["inp_norm", "in0", "Qcur", "Kcur", "Vcur", "Rcur", "kqv_gated", "attn_out", "ffn_inp", "ffn_out", "l_out"]:
    fq = f"{q}.{name}-{slot}.f32"; fr = f"{r}.{name}-{slot}.f32"
    if not os.path.exists(fq) or not os.path.exists(fr): print(f"{name:10s} (missing {'q' if not os.path.exists(fq) else 'r'})"); continue
    a = np.fromfile(fq, np.float32); b = np.fromfile(fr, np.float32)
    if a.size != b.size: print(f"{name:10s} size {a.size} vs {b.size}"); continue
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"{name:10s} cos {c:.6f}  |q| {np.linalg.norm(a):9.4f} |r| {np.linalg.norm(b):9.4f}  q[:4] {a[:4]} r[:4] {b[:4]}")
