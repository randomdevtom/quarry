# usage: python cache_sim.py model.gguf route_trace.txt   (needs numpy + gguf)
import sys, collections
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
exp = sum(int(t.n_bytes) for t in r.tensors if 'exps' in t.name)
emb = sum(int(t.n_bytes) for t in r.tensors if t.name == 'token_embd.weight')
dense = sum(int(t.n_bytes) for t in r.tensors if 'exps' not in t.name) - emb
L, E, K = 48, 128, 8
esz = exp / (L * E)
print(f"experts {exp/1e9:.2f} GB ({esz/1e6:.2f} MB each), dense {dense/1e9:.2f} GB, embd {emb/1e9:.2f} GB")

trace = collections.defaultdict(list)  # pos -> [(layer, experts)]
for line in open(sys.argv[2] if len(sys.argv) > 2 else 'route_trace.txt'):
    p = list(map(int, line.split()))
    trace[p[0]].append((p[1], p[2:]))
positions = sorted(trace)
PROMPT = 26
seq = [[(l, e) for l, es in sorted(trace[p]) for e in es] for p in positions]
gen = [s for p, s in zip(positions, seq) if p >= PROMPT]
warm = [s for p, s in zip(positions, seq) if p < PROMPT]
# per-token reuse: fraction of experts used at t that were used at t-1 (same layer)
reuse = []
for a, b in zip(gen, gen[1:]):
    reuse.append(len(set(a) & set(b)) / len(b))
print(f"consecutive-token expert reuse: {sum(reuse)/len(reuse)*100:.1f}%")
uniq = len({x for s in gen for x in s}); print(f"distinct experts touched in {len(gen)} gen tokens: {uniq}/{L*E}")

def sim(cap, policy):
    cache = collections.OrderedDict(); freq = collections.Counter(); hits = miss = 0
    future = None
    if policy == 'belady':
        flat = [x for s in warm + gen for x in s]
        nxt = {}; future = [0] * len(flat)
        for i in range(len(flat) - 1, -1, -1):
            future[i] = nxt.get(flat[i], 1 << 30); nxt[flat[i]] = i
    i = 0
    for phase, toks in (('warm', warm), ('gen', gen)):
        for s in toks:
            for x in s:
                freq[x] += 1
                if x in cache:
                    if phase == 'gen': hits += 1
                    if policy == 'lru': cache.move_to_end(x)
                    if policy == 'belady': cache[x] = future[i]
                else:
                    if phase == 'gen': miss += 1
                    if len(cache) >= cap:
                        if policy == 'lru': cache.popitem(last=False)
                        elif policy == 'lfu': del cache[min(cache, key=lambda k: freq[k])]
                        else: del cache[max(cache, key=lambda k: cache[k])]
                    cache[x] = future[i] if policy == 'belady' else None
                i += 1
    return hits / (hits + miss)

ram_for_experts = [4e9, 5e9, 6e9, 7e9]
print(f"\n{'expert RAM':>10} {'frac':>5} {'LRU':>6} {'LFU':>6} {'oracle':>7}   est tok/s (LFU, no overlap / full overlap)")
for rb in ram_for_experts:
    cap = int(rb / esz)
    h = {p: sim(cap, p) for p in ('lru', 'lfu', 'belady')}
    per_tok_exp = L * K * esz
    t_ram = (dense + per_tok_exp) / 23e9
    t_ssd = (1 - h['lfu']) * per_tok_exp / 0.95e9
    print(f"{rb/1e9:>9.0f}G {cap/(L*E):>5.0%} {h['lru']:>6.1%} {h['lfu']:>6.1%} {h['belady']:>7.1%}   "
          f"{1/(t_ram+t_ssd):5.1f} / {1/max(t_ram,t_ssd):5.1f}")
