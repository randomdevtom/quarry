#!/usr/bin/env python3
"""HumanEval subset through a running `quarry serve` (OpenAI API): pass@1, greedy.

usage: code_bench.py --name qwen --port 8090 [--every 8] [--max-tokens 768]
Writes results/<name>.jsonl next to this file and prints a summary.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("HUMANEVAL", os.path.join(HERE, "data", "HumanEval.jsonl"))
INSTRUCTION = ("Complete the following Python function. "
               "Reply with the full function in a single ```python code block.\n\n")


def ask(port, prompt, max_tokens):
    body = json.dumps({"model": "quarry", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("usage", {}).get("completion_tokens", 0)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def run_tests(task, code, timeout=10):
    program = (task["prompt"] + "\n    pass\n\n" + code + "\n\n" + task["test"] +
               f"\n\ncheck({task['entry_point']})\n")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, "" if r.returncode == 0 else (r.stderr.strip().splitlines() or ["?"])[-1][:200]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(DATA)][::a.every]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", a.name + ".jsonl")
    done = {}
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done[r["task_id"]] = r
    with open(out, "a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            t0 = time.time()
            text, ntok = ask(a.port, INSTRUCTION + t["prompt"], a.max_tokens)
            ok, err = run_tests(t, extract_code(text))
            r = {"task_id": t["task_id"], "passed": ok, "error": err, "tokens": ntok,
                 "seconds": round(time.time() - t0, 1), "answer": text}
            f.write(json.dumps(r) + "\n"); f.flush()
            done[t["task_id"]] = r
            print(f"{a.name} {t['task_id']:14s} {'PASS' if ok else 'fail'} {ntok:4d} tok {r['seconds']:6.1f}s {err[:60]}", flush=True)
    rows = [done[t["task_id"]] for t in tasks if t["task_id"] in done]
    p = sum(r["passed"] for r in rows)
    print(f"== {a.name}: pass {p}/{len(rows)} ({100 * p / len(rows):.0f}%)  avg {sum(r['tokens'] for r in rows) / len(rows):.0f} tokens, "
          f"{sum(r['seconds'] for r in rows) / len(rows):.0f}s per task")


if __name__ == "__main__":
    main()
