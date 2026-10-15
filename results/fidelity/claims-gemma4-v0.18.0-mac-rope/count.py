#!/usr/bin/env python3
"""Recount the #3913 rotary-position (rope) fp16 raw results of the Mac run (stdlib only; reads only).

Reads raw/<cases>__<model>-<file>__<backend>-<precision>.jsonl (PowerBeef's runner, one line per case),
raw/recheck/*.jsonl, the own-suite runs ../claims-gemma-4-*-v0.18.0-mac-*.jsonl, logs/*.log (engine logs)
and, when given, PowerBeef's published M6 results (his research/issue-3913 folder: cases/ for the
line counts, results/mac/ for the comparison rows). Exactness is recomputed from `output` and
`expected` (ASCII digits only) and checked against the runner's `exact`; `expected` is checked
against the runner's LCG (referenceDigits) recomputed here.

    python3 -I count.py [raw dir] [research/issue-3913 dir] > count.txt
    (defaults: raw/ next to this script; the second argument or G3913_UPSTREAM for his folder)
"""
import glob, json, os, re, statistics, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))          # results/fidelity/claims-gemma4-v0.18.0-mac-rope
RAW = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "raw")
UP = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("G3913_UPSTREAM", "")
SUITE_GLOB = os.path.join(os.path.dirname(HERE), "claims-gemma-4-*-v0.18.0-mac-*.jsonl")
SUITE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "evaldata", "fidelity_suite_v1.jsonl")
LOGS = os.path.join(HERE, "logs")
CASES_ORDER = ["short", "baseline", "long", "cross4096", "long3", "xl"]
FILES = ["original", "rope2048", "rope512"]
CONFIGS = ["gpu-fp16", "gpu-fp32", "cpu-fp32"]
DIG = set("0123456789")


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def ref_digits(seed):
    m, a, c = (1 << 64) - 1, 6364136223846793005, 1442695040888963407
    s = (seed * a + c) & m
    out = []
    for _ in range(48):
        s = (s * a + c) & m
        out.append(str((s >> 33) % 10))
    return "".join(out)


def exact(r):
    return "".join(ch for ch in r["output"] if ch in DIG) == r["expected"]


def frac(rows):
    return f"{sum(exact(r) for r in rows)}/{len(rows)}"


# ------------------------------------------------------------------ load + integrity
arms = {}  # (cases, model, file, config) -> {id: row}
problems = []
for path in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
    base = os.path.basename(path)[:-6]
    try:
        cases, label, config = base.split("__")
        model, file = label.split("-", 1)
    except ValueError:
        problems.append(f"unparsed file name {base}")
        continue
    rows = load(path)
    if UP:
        want = len(json.load(open(os.path.join(UP, "cases", f"{cases}.json"))))
        if len(rows) != want:
            problems.append(f"{base}: {len(rows)} lines, cases file has {want}")
    for r in rows:
        if r["expected"] != ref_digits(r["seed"]):
            problems.append(f"{base} {r['id']}: expected != referenceDigits(seed)")
        if exact(r) != r["exact"]:
            problems.append(f"{base} {r['id']}: recomputed exact {exact(r)} != runner {r['exact']}")
        if not r["exact"] and r.get("firstMismatchPosition") != r["prefill"] - 1 + r["firstMismatch"]:
            problems.append(f"{base} {r['id']}: firstMismatchPosition inconsistent")
        if r["backend"] + "-" + r["activation"] != config:
            problems.append(f"{base} {r['id']}: record backend/activation {r['backend']}/{r['activation']} != {config}")
    arms[(cases, model, file, config)] = {r["id"]: r for r in rows}

print("# count.py — #3913 rope fp16 (M4 Max), recount of", os.path.relpath(RAW, HERE) + "/"
      + ("" if UP else " (PowerBeef's folder not given: line counts against his cases/ and his rows skipped)"))
print(f"arms {len(arms)}, replies {sum(len(v) for v in arms.values())}")
print("integrity:", ("OK (" + ("line counts, " if UP else "") + "expected = LCG(seed), recomputed exact = runner exact, firstMismatchPosition)") if not problems else "")
for p in problems:
    print("  PROBLEM", p)

# prefill identity: the same case id must have the same prefill in every arm (same prompt, same template)
pre = defaultdict(set)
for (cases, model, file, config), rows in arms.items():
    for i, r in rows.items():
        pre[(cases, i)].add(r["prefill"])
bad = {k: v for k, v in pre.items() if len(v) > 1}
print(f"prefill identity across arms: {len(pre) - len(bad)}/{len(pre)} case ids have one prefill value"
      + ("" if not bad else f"; differing: {sorted(bad.items())[:10]}"))
pb = {}
for path in glob.glob(os.path.join(UP, "results", "mac", "*__original__*.jsonl")) if UP else []:
    cases = os.path.basename(path).split("__")[0]
    for r in load(path):
        pb[(cases, r["id"])] = r["prefill"]
if pb:
    same = [k for k in pre if k in pb and pre[k] == {pb[k]}]
    diff = [k for k in pre if k in pb and pre[k] != {pb[k]}]
    print(f"prefill vs PowerBeef's M6 original records (same case id): {len(same)} equal, {len(diff)} differ"
          + ("" if not diff else f" {[(k, sorted(pre[k]), pb[k]) for k in diff[:6]]}"))

# ------------------------------------------------------------------ table 1
present = [c for c in CASES_ORDER if any(k[0] == c for k in arms)]
print("\n## 1. exact / total (M4 Max, LiteRT-LM v0.18.0, PowerBeef's runner and cases)\n")
print("| model | file | backend-precision | " + " | ".join(f"`{c}`" for c in present) + " |")
print("|---|---|---|" + "---|" * len(present))
for model in ["E2B", "E4B"]:
    for file in FILES:
        for config in CONFIGS:
            cells = [frac(arms[(c, model, file, config)].values()) if (c, model, file, config) in arms else "–" for c in present]
            if any(x != "–" for x in cells):
                print(f"| {model} | {file} | {config} | " + " | ".join(cells) + " |")
for c in present:
    ps = [r["prefill"] for k, v in arms.items() if k[0] == c for r in v.values()]
    print(f"- `{c}`: prompts {min(ps)}–{max(ps)} tokens (engine prefill count)")

# PowerBeef's M6 rows from his raw (E4B only; his "split" = rope2048), same cells
pbarms = {}
for path in sorted(glob.glob(os.path.join(UP, "results", "mac", "*.jsonl"))) if UP else []:
    cases, lab, config = os.path.basename(path)[:-6].split("__")
    if lab in ("original", "split") and cases in present:
        pbarms[(cases, "rope2048" if lab == "split" else lab, config)] = load(path)
if pbarms:
    print("\nPowerBeef, Apple M6 (his raw `results/mac/`, E4B; his `split` file = rope2048 here; fp16 records say activation `default`):\n")
    print("| model | file | backend-precision | " + " | ".join(f"`{c}`" for c in present) + " |")
    print("|---|---|---|" + "---|" * len(present))
    for file in ["original", "rope2048"]:
        for config in CONFIGS:
            cells = [frac(pbarms[(c, file, config)]) if (c, file, config) in pbarms else "–" for c in present]
            if any(x != "–" for x in cells):
                print(f"| E4B (M6) | {file} | {config} | " + " | ".join(cells) + " |")

# ------------------------------------------------------------------ table 2
GROUPS = [("1704–1751", 1751), ("1904–1951 (incl. late)", 1951), ("1952–2044", 2044), ("2011–2638", 10 ** 9)]


def group_of(r):
    end = r["prefill"] + 46  # last digit's position; first digit at prefill - 1
    for name, hi in GROUPS:
        if end <= hi:
            return name


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[group_of(r)].append(r)
    return g


print("\n## 2. `baseline` by reply digit positions (prefill − 1 … prefill + 46), exact / total\n")
print("| model | file | backend-precision | " + " | ".join(n for n, _ in GROUPS) + " |")
print("|---|---|---|---|---|---|---|")
for model in ["E2B", "E4B"]:
    for file in FILES:
        for config in CONFIGS:
            rows = arms.get(("baseline", model, file, config))
            if rows:
                g = groups(rows.values())
                print(f"| {model} | {file} | {config} | " + " | ".join(frac(g[n]) if g[n] else "–" for n, _ in GROUPS) + " |")
pbase = os.path.join(UP, "results", "mac", "baseline__original__gpu-fp16.jsonl") if UP else ""
if pbase and os.path.exists(pbase):
    prow = load(pbase)
    g = groups(prow)
    print("| E4B (PowerBeef M6, his raw) | original | gpu-fp16 | " + " | ".join(frac(g[n]) if g[n] else "–" for n, _ in GROUPS) + " |")
anyrows = next((v for k, v in arms.items() if k[0] == "baseline"), None)
if anyrows:
    g = groups(anyrows.values())
    for n, _ in GROUPS:
        rs = g[n]
        print(f"- group {n}: {len(rs)} cases, positions {min(r['prefill'] for r in rs) - 1}–{max(r['prefill'] for r in rs) + 46},"
              f" prefill {sorted({r['prefill'] for r in rs})}, late {sum(r['late'] for r in rs)}")
    mine = arms.get(("baseline", "E4B", "original", "gpu-fp16"))
    if mine and pbase and os.path.exists(pbase):
        his = {r["id"]: r for r in prow}
        agree = [i for i in mine if i in his and exact(mine[i]) == exact(his[i])]
        dis = sorted(i for i in mine if i in his and exact(mine[i]) != exact(his[i]))
        print(f"- E4B original fp16 `baseline`, M4 Max vs M6 (PowerBeef): same pass/fail in {len(agree)}/{len(mine)};"
              f" differ: {[(i, 'M4Max ' + str(exact(mine[i])), 'M6 ' + str(exact(his[i]))) for i in dis]}")

# ------------------------------------------------------------------ 3. null test + text identity
print("\n## 3. text identity\n")
for model in ["E2B", "E4B"]:
    for file in ["rope2048", "rope512"]:
        for cases in present:
            a, b = arms.get((cases, model, file, "gpu-fp32")), arms.get((cases, model, "original", "gpu-fp32"))
            if a and b:
                keys = [k for k in b if k in a]
                print(f"- null test {model} {file} fp32 vs original fp32, `{cases}`: identical text {sum(a[k]['output'] == b[k]['output'] for k in keys)}/{len(keys)}")
for model in ["E2B", "E4B"]:
    for file in FILES:
        for cases in present:
            a, b = arms.get((cases, model, file, "gpu-fp16")), arms.get((cases, model, "original", "gpu-fp32"))
            if a and b:
                keys = [k for k in b if k in a]
                print(f"- {model} {file} fp16 vs original fp32, `{cases}`: identical text {sum(a[k]['output'] == b[k]['output'] for k in keys)}/{len(keys)}")
for model in ["E2B", "E4B"]:
    for file in FILES:
        for cases in present:
            a, b = arms.get((cases, model, file, "gpu-fp16")), arms.get((cases, model, file, "gpu-fp32"))
            if a and b:
                keys = [k for k in b if k in a]
                print(f"- {model} {file} fp16 vs {file} fp32 (same file), `{cases}`: text differs in {sum(a[k]['output'] != b[k]['output'] for k in keys)}/{len(keys)}")
for model in ["E2B", "E4B"]:
    a, b = arms.get(("long3", model, "original", "cpu-fp32")), arms.get(("long", model, "original", "gpu-fp32"))
    if a and b:
        keys = [k for k in a if k in b]
        print(f"- {model} original cpu fp32 vs gpu fp32 (`long3` ids in `long`): identical text {sum(a[k]['output'] == b[k]['output'] for k in keys)}/{len(keys)}")

# ------------------------------------------------------------------ 4. boundary
print("\n## 4. failures by position (gpu-fp16)\n")
for model in ["E2B", "E4B"]:
    for file in FILES:
        allf = []
        for cases in present:
            rows = arms.get((cases, model, file, "gpu-fp16"))
            if not rows:
                continue
            fails = [r for r in rows.values() if not exact(r)]
            allf += fails
            if fails:
                pos = sorted(r["firstMismatchPosition"] for r in fails)
                extra = ""
                if file != "original":
                    blk = 2048 if file == "rope2048" else 512
                    mod = sorted(p % blk for p in pos)
                    extra = f"; position mod {blk} {mod[0]}–{mod[-1]}"
                below = [r["id"] for r in fails if r["prefill"] + 46 < 2048]
                print(f"- {model} {file} `{cases}`: {len(fails)} fail, firstMismatchPosition {pos[0]}–{pos[-1]}{extra}"
                      f"; prefill {min(r['prefill'] for r in fails)}–{max(r['prefill'] for r in fails)}"
                      + (f"; all digits below 2048 in {below}" if below else ""))
            else:
                print(f"- {model} {file} `{cases}`: 0 fail")
        oks = [r for cases in present for r in (arms.get((cases, model, file, "gpu-fp16")) or {}).values() if exact(r)]
        if allf:
            print(f"  = {model} {file}: min firstMismatchPosition {min(r['firstMismatchPosition'] for r in allf)}"
                  f" ({min(allf, key=lambda r: r['firstMismatchPosition'])['id']}),"
                  f" highest last-digit position among exact replies {max((r['prefill'] + 46 for r in oks), default='-')}")
for model in ["E2B", "E4B"]:
    rows = arms.get(("baseline", model, "original", "gpu-fp16"))
    if rows:
        low = [r for r in sorted(rows.values(), key=lambda r: r["prefill"]) if not exact(r) and r["prefill"] + 46 < 2048]
        print(f"- {model} original fp16 `baseline`, failures whose digits are all below 2048: "
              + (", ".join(f"{r['id']} (prefill {r['prefill']}, first error {r['firstMismatchPosition']}, output {r['output']!r})" for r in low) or "none"))

# ------------------------------------------------------------------ 5. speed
print("\n## 5. runner-recorded prefill / decode tok/s, medians (recorded by the runner, not a speed benchmark)\n")
for cases in ["long", "xl"]:
    for model in ["E2B", "E4B"]:
        for file in FILES:
            for config in CONFIGS:
                rows = arms.get((cases, model, file, config))
                if rows:
                    v = [r for r in rows.values() if "prefillTPS" in r]
                    if v:
                        print(f"- `{cases}` {model} {file} {config}: prefill {statistics.median(r['prefillTPS'] for r in v):.0f},"
                              f" decode {statistics.median(r['decodeTPS'] for r in v):.1f}, seconds/reply {statistics.median(r['seconds'] for r in v):.2f} (n={len(v)})")

# ------------------------------------------------------------------ 3a. replies equal to another case's reference
allref = {}
for (cases, model, file, config), rows in arms.items():
    for i, r in rows.items():
        allref.setdefault(r["expected"], set()).add(r["seed"])
odd = []
for (cases, model, file, config), rows in arms.items():
    # the arm's case order: his cases file when given, else the record order (the runner writes the cases in order)
    order = [c["id"] for c in json.load(open(os.path.join(UP, "cases", f"{cases}.json")))] if UP else list(rows)
    for i, r in rows.items():
        got = "".join(ch for ch in r["output"] if ch in DIG)
        if not exact(r) and got in allref:
            prev = order[order.index(i) - 1] if order.index(i) > 0 else None
            odd.append(f"{cases}__{model}-{file}__{config} {i}: digits = reference of seed {sorted(allref[got])}; previous case in the arm {prev}")
print(f"\n## 3a. failed replies whose digits equal another case's reference: {len(odd)} of {sum(len(v) for v in arms.values())} replies\n")
for o in odd:
    print("- " + o)

# ------------------------------------------------------------------ 3b. recheck
rc = sorted(glob.glob(os.path.join(RAW, "recheck", "*.jsonl")))
if rc:
    print("\n## 3b. recheck: E4B original gpu fp32, `s257-seed4` (raw/recheck/)\n")
    by_seed = {s: ref_digits(s) for s in range(1, 6)}
    for path in rc:
        for r in load(path):
            got = "".join(ch for ch in r["output"] if ch in DIG)
            same = [s for s, d in by_seed.items() if d == got]
            print(f"- {os.path.basename(path)[:-6]}: {r['id']} prefill {r['prefill']} exact {exact(r)}"
                  + (f", output = reference digits of seed {same}" if same and not exact(r) else "")
                  + ("" if exact(r) else f", firstMismatch {r.get('firstMismatch')}"))

# ------------------------------------------------------------------ 6. own suite
if glob.glob(SUITE_GLOB):
    suite = {json.loads(l)["id"]: json.loads(l) for l in open(SUITE)}
    print("\n## 6. own fidelity suite (verbatim-substring pass test, recounted here)\n")
    print("| file | model | backend | version | reps | pass / gens | failing cases / cases | failing by bucket | max distinct outputs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(SUITE_GLOB)):
        rows = load(path)
        if not rows:
            continue
        ok = lambda r: not r.get("error") and all(v in r["output"] for v in suite[r["case_id"]]["expected"].values())
        ids = sorted({r["case_id"] for r in rows})
        fails = sorted({r["case_id"] for r in rows if not ok(r)})
        nb = Counter(suite[i]["context_bucket"] for i in ids)
        fb = Counter(suite[i]["context_bucket"] for i in fails)
        distinct = max(len({r["output"] for r in rows if r["case_id"] == i}) for i in ids)
        print(f"| {os.path.basename(path)} | {sorted({r['model'] for r in rows})} | {sorted({r['backend'] for r in rows})}"
              f" | {sorted({r.get('runtime_version') for r in rows})} | {min(r['rep'] for r in rows)}–{max(r['rep'] for r in rows)}"
              f" | {sum(ok(r) for r in rows)}/{len(rows)} | {len(fails)}/{len(ids)}"
              f" | {', '.join(f'{b}: {fb.get(b, 0)}/{nb[b]}' for b in sorted(nb))} | {distinct} |")
        for c in fails:
            got = next(r["output"] for r in rows if r["case_id"] == c and not ok(r))
            pt = sorted({r.get("prefill_tokens") for r in rows if r["case_id"] == c})
            print(f"  - FAIL `{c}` bucket {suite[c]['context_bucket']} prefill {pt}: expected {suite[c]['expected']} got {got[:100]!r}")

# ------------------------------------------------------------------ 7. engine identity
print("\n## 7. engine log lines (verbatim, deduplicated; count = logs containing the line)\n")
pat = re.compile(r"Selected adapter|delegate_webgpu|registered|[Vv]ersion|RegisterAccelerator|backend=|WebGPU|Metal|OpenCL|xnnpack|XNNPACK|engine loaded|WATCHDOG")
seen = Counter()
for path in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
    lines = set()
    for l in open(path, errors="replace"):
        if pat.search(l):
            l = re.sub(r"^[IWE]\d{4} \S+ \d+ ", "", l.strip())          # absl prefix: level+date, time, thread
            l = re.sub(r"ptr=0x[0-9a-f]+", "ptr=…", re.sub(r"\d+\.\d+ s$", "<t> s", l))
            lines.add(l[:220])
    for l in lines:
        seen[l] += 1
for l, n in sorted(seen.items(), key=lambda x: (-x[1], x[0]))[:40]:
    print(f"- ({n}) {l}")
