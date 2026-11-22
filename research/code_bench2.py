#!/usr/bin/env python3
"""Coding benchmarks through a running `quarry serve` (OpenAI API), greedy pass@1.

  HumanEval   164 tasks, original unit tests
  HumanEval+  same answers, EvalPlus inputs (base + ~1000 extra per task) checked
              against the canonical solution's outputs
  MBPP        sanitized test split (257 tasks), its assert tests

usage: code_bench2.py --name qwen --port 8090 --bench humaneval|mbpp [--limit N]
Results: results/<name>.<bench>.jsonl (resumable), summary printed at the end.

Data (in research/data/, or set HUMANEVAL / MBPP):
  HumanEval.jsonl       https://github.com/openai/human-eval (data/HumanEval.jsonl.gz, gunzipped)
  HumanEvalPlus.jsonl.gz https://github.com/evalplus/humanevalplus_release/releases (v0.1.10)
  mbpp.jsonl            sanitized MBPP, one task per line: task_id, text, test_list, test_imports
Generated code is executed: run this only in a sandbox you trust.
"""
import argparse, gzip, json, os, re, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HE = os.environ.get("HUMANEVAL", os.path.join(HERE, "data", "HumanEval.jsonl"))
HEP = os.path.join(HERE, "data", "HumanEvalPlus.jsonl.gz")
MBPP = os.environ.get("MBPP", os.path.join(HERE, "data", "mbpp.jsonl"))
HE_INSTR = ("Complete the following Python function. "
            "Reply with the full function in a single ```python code block.\n\n")


def ask(port, prompt, max_tokens):
    body = json.dumps({"model": "quarry", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("usage", {}).get("completion_tokens", 0)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def run_py(program, timeout):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return True, ""
            return False, (r.stderr.strip().splitlines() or ["?"])[-1][:200]
        except subprocess.TimeoutExpired:
            return False, "timeout"


# EvalPlus-style check: run candidate and canonical on every input, compare outputs
PLUS_RUNNER = r'''
import json, copy, math, sys
sys.setrecursionlimit(10000)
def same(a, b, atol):
    if isinstance(a, float) or isinstance(b, float):
        try:
            if math.isnan(a) and math.isnan(b): return True
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=max(atol, 1e-9))
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, atol) for x, y in zip(a, b))
    return a == b
ns_ref, ns_cand = {}, {}
exec(REF, ns_ref)
exec(CAND, ns_cand)
ref, cand = ns_ref[ENTRY], ns_cand[ENTRY]
for i, inp in enumerate(INPUTS):
    try:
        exp = ref(*copy.deepcopy(inp))
    except Exception:
        continue                    # input the reference rejects (contract)
    got = cand(*copy.deepcopy(inp))
    if ENTRY == "find_zero":        # any root is fine: check the polynomial value, as EvalPlus does
        xs = inp[0]
        scale = max(1.0, sum(abs(c) * abs(got) ** k for k, c in enumerate(xs)))   # relative to term size
        pv = lambda x: abs(sum(c * x ** k for k, c in enumerate(xs)))
        # as close to a root as the reference answer (10x slack), or within tolerance
        if pv(got) > max(10 * pv(exp), max(ATOL, 1e-6) * scale):
            raise AssertionError(f"input {i}: poly({got}) != 0")
        continue
    if not same(got, exp, ATOL):
        raise AssertionError(f"input {i}: got {str(got)[:60]} expected {str(exp)[:60]}")
'''


def humaneval_tasks():
    plus = {r["task_id"]: r for r in (json.loads(l) for l in gzip.open(HEP, "rt"))}
    for l in open(HE):
        t = json.loads(l)
        t["plus"] = plus[t["task_id"]]
        yield t


def eval_humaneval(t, code):
    base = t["prompt"] + "\n    pass\n\n" + code + "\n\n" + t["test"] + f"\n\ncheck({t['entry_point']})\n"
    ok, err = run_py(base, 20)
    p = t["plus"]
    cand = t["prompt"] + "\n    pass\n\n" + code
    ref = p["prompt"] + p["canonical_solution"]
    prog = (f"REF = {ref!r}\nCAND = {cand!r}\nENTRY = {t['entry_point']!r}\nATOL = {float(p.get('atol') or 0)!r}\n"
            f"INPUTS = json_inputs = __import__('json').loads({json.dumps(p['base_input'] + p['plus_input'])!r})\n" + PLUS_RUNNER)
    okp, errp = run_py(prog, 120) if ok else (False, "base failed")
    return {"passed": ok, "error": err, "passed_plus": okp, "error_plus": errp}


def mbpp_tasks():
    for l in open(MBPP):
        t = json.loads(l)
        if 11 <= int(t["task_id"].split("/")[1]) <= 510:
            yield t


def mbpp_prompt(t):
    return (f"{t['text']}\nYour code should pass these tests:\n\n" + "\n".join(t["test_list"]) +
            "\n\nReply with the Python code in a single ```python code block.")


def eval_mbpp(t, code):
    prog = "\n".join(t.get("test_imports") or []) + "\n" + code + "\n\n" + "\n".join(t["test_list"]) + "\n"
    ok, err = run_py(prog, 20)
    return {"passed": ok, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--bench", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=768)
    a = ap.parse_args()
    tasks = list(humaneval_tasks() if a.bench == "humaneval" else mbpp_tasks())
    if a.limit:
        tasks = tasks[:a.limit]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{a.name}.{a.bench}.jsonl")
    done = {}
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done[r["task_id"]] = r
    with open(out, "a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            t0 = time.time()
            prompt = HE_INSTR + t["prompt"] if a.bench == "humaneval" else mbpp_prompt(t)
            text, ntok = ask(a.port, prompt, a.max_tokens)
            code = extract_code(text)
            res = eval_humaneval(t, code) if a.bench == "humaneval" else eval_mbpp(t, code)
            r = {"task_id": t["task_id"], **res, "tokens": ntok, "seconds": round(time.time() - t0, 1), "answer": text}
            f.write(json.dumps(r) + "\n"); f.flush()
            done[t["task_id"]] = r
            extra = f" plus {'PASS' if res.get('passed_plus') else 'fail'}" if a.bench == "humaneval" else ""
            print(f"{a.bench} {t['task_id']:14s} {'PASS' if res['passed'] else 'fail'}{extra} {ntok:4d} tok {r['seconds']:6.1f}s", flush=True)
    rows = [done[t["task_id"]] for t in tasks if t["task_id"] in done]
    n = len(rows)
    p = sum(r["passed"] for r in rows)
    msg = f"== {a.name} {a.bench}: pass@1 {p}/{n} ({100 * p / n:.1f}%)"
    if a.bench == "humaneval":
        pp = sum(r.get("passed_plus", False) for r in rows)
        msg += f"  HumanEval+ {pp}/{n} ({100 * pp / n:.1f}%)"
    msg += f"  avg {sum(r['tokens'] for r in rows) / n:.0f} tokens, {sum(r['seconds'] for r in rows) / n:.0f}s/task"
    print(msg)


if __name__ == "__main__":
    main()
