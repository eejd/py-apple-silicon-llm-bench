#!/usr/bin/env python3
"""count.py — LiteRT-LM #3913 on a Pixel 8a: the tables from the raw records only (../raw/*.jsonl), stdlib only.

    python3 -I count.py [raw dir] > ../count.txt    # prints the tables (default raw dir: ../raw, or G3913_RAW_DIR)

Tables: (1) model x backend exact/total over the 26-question subset (and the CPU controls); (2) per model
on the GPU, the reply's digit positions (prefill - 1 ... prefill + 46) grouped two ways: by case file (`baseline` 18 /
`long` 8, with the position range computed from the records) and by the 2048 boundary (every digit below 2048 / the
reply crosses 2048 / every digit at 2048 or above); (3) CPU controls; (4) prefill = the Mac's for the same question;
(5) the first mismatch positions of the original GPU rows that failed; (6) the CLI's process wall, median, cold and warm
apart (not a speed benchmark: engine load, program cache, prefill and decode in one number); (7) the GPU's identity
lines, verbatim from the first GPU leg's stderr, the runtime and the device line of the setup log. Rows whose status is
not DONE are counted apart (never as inexact). Every number is read from a record; the only arithmetic is counting and
a median.
"""
import glob
import json
import os
import re
import statistics
import sys

KIT = os.path.dirname(os.path.abspath(__file__))      # .../claims-gemma4-pixel8a-rope/repro
BASE = os.path.dirname(KIT)                           # .../claims-gemma4-pixel8a-rope
RAW = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.environ.get("G3913_RAW_DIR") or os.path.join(BASE, "raw")
MAN = json.load(open(os.path.join(KIT, "prompts", "manifest.json")))
VARIANTS = ["original", "rope512", "rope2048"]


def load():
    recs = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                d["_file"] = os.path.relpath(f, BASE)
                recs.append(d)
    return recs


def frac(rows):
    ok = [r for r in rows if r["status"] == "DONE"]
    other = len(rows) - len(ok)
    return f"{sum(1 for r in ok if r['exact'])}/{len(ok)}" + (f" (+{other} not DONE)" if other else "")


def span(rows):
    ok = [r for r in rows if r.get("prefill")]
    if not ok:
        return "-"
    return f"{min(r['prefill'] - 1 for r in ok)}–{max(r['prefill'] + 46 for r in ok)}"


def side(r):
    lo, hi = r["prefill"] - 1, r["prefill"] + 46
    return "below 2048" if hi < 2048 else "at/above 2048" if lo >= 2048 else "crosses 2048"


def main():
    recs = load()
    out = []
    p = out.append
    p(f"# #3913 Pixel 8a counts (repro/count.py) from {len(recs)} records in {os.path.relpath(RAW, BASE)}/")
    by = {}
    for r in recs:
        by.setdefault((r["variant"], r["backend"], r["precision"]), []).append(r)
    p("\n## (1) model x backend: exact / total (DONE rows), the 26-question subset; other questions apart")
    p("| model | backend | activation | subset 26 | other questions | files |")
    p("|---|---|---|---|---|---|")
    for (v, be, pr), rows in sorted(by.items(), key=lambda x: (VARIANTS.index(x[0][0]) if x[0][0] in VARIANTS else 9, x[0][1], x[0][2])):
        sub = [r for r in rows if "gpu26" in MAN[r["id"]]["subset"]]
        oth = [r for r in rows if "gpu26" not in MAN[r["id"]]["subset"]]
        p(f"| {v} | {be} | {pr} | {frac(sub) if sub else '-'} | {frac(oth) if oth else '-'} | "
          f"{', '.join(sorted({r['_file'].split('/')[-1] for r in rows}))} |")
    p("\n## (2) GPU, the subset by group: exact / total (digit positions prefill-1 … prefill+46 from the records)")
    p("| model | baseline 18 | positions | long 8 | positions | below 2048 | crosses 2048 | at/above 2048 |")
    p("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        rows = [r for r in by.get((v, "gpu", "fp16"), []) if "gpu26" in MAN[r["id"]]["subset"] and r.get("prefill")]
        if not rows:
            continue
        b = [r for r in rows if MAN[r["id"]]["cases_file"] == "baseline"]
        l = [r for r in rows if MAN[r["id"]]["cases_file"] == "long"]
        g = {s: [r for r in rows if side(r) == s] for s in ("below 2048", "crosses 2048", "at/above 2048")}
        p(f"| {v} | {frac(b)} | {span(b)} | {frac(l)} | {span(l)} | {frac(g['below 2048'])} | {frac(g['crosses 2048'])} | "
          f"{frac(g['at/above 2048'])} |")
    p("\n## (3) CPU controls")
    for (v, be, pr), rows in sorted(by.items()):
        if be == "cpu":
            p(f"- {v} cpu {pr}: {frac(rows)} — " + ", ".join(f"{r['id']} {'exact' if r['exact'] else 'INEXACT'}" for r in rows))
    p("\n## (4) prefill = the Mac's for the same question")
    m = [r for r in recs if r.get("prefill_match") is not None]
    p(f"- {sum(1 for r in m if r['prefill_match'])}/{len(m)} records match "
      f"(reference: {', '.join(sorted({r['prefill_mac_src'] for r in m if r.get('prefill_mac_src')}))})")
    for r in m:
        if not r["prefill_match"]:
            p(f"  - DIFF {r['tag']}: android {r['prefill']} mac {r['prefill_mac']}")
    turns = sorted({len(r.get("prefill_turns") or []) for r in recs if r["status"] == "DONE"})
    p(f"- prefill turns per process: {turns} (the record's prefill = the sum of the turns; with one turn it is the "
      "runner's lastPrefillTokenCount)")
    p("\n## (5) original GPU rows that failed: first mismatch position")
    bad = [r for r in by.get(("original", "gpu", "fp16"), []) if r["status"] == "DONE" and not r["exact"]]
    if bad:
        pos = [r["firstMismatchPosition"] for r in bad]
        p(f"- {len(bad)} rows, firstMismatchPosition min {min(pos)} max {max(pos)}")
        for r in sorted(bad, key=lambda r: r["prefill"]):
            p(f"  - {r['id']}: prefill {r['prefill']}, first mismatch at digit {r['firstMismatch']} = position "
              f"{r['firstMismatchPosition']}, decode {r['decode']}, output {r['output'][:70]}")
    else:
        p("- none")
    for v in ("rope512", "rope2048"):
        bb = [r for r in by.get((v, "gpu", "fp16"), []) if r["status"] == "DONE" and not r["exact"]]
        if bb:
            p(f"- {v} GPU rows that failed: " + "; ".join(f"{r['id']} pos {r['firstMismatchPosition']}" for r in bb))
    p("\n## (6) the CLI process wall on the phone, median s (cold / warm; not a speed benchmark)")
    for (v, be, pr), rows in sorted(by.items()):
        c = [r["wall_s"] for r in rows if r.get("wall_s") is not None and r["cold"]]
        w = [r["wall_s"] for r in rows if r.get("wall_s") is not None and not r["cold"]]
        p(f"- {v} {be} {pr}: cold {statistics.median(c) if c else '-'} (n {len(c)}), warm {statistics.median(w) if w else '-'} "
          f"(n {len(w)}, min {min(w) if w else '-'} max {max(w) if w else '-'})")
    p("\n## (7) the GPU's identity (first GPU record's stderr lines, verbatim), runtime, device")
    g = [r for r in recs if r["backend"] == "gpu" and r["status"] == "DONE"]
    if g:
        e = g[0]["engine"]
        for k in ("registered", "device", "delegate", "activation", "weights"):
            for line in e.get(k) or []:
                p(f"- {k}: `{line}`")
        p(f"- runtime: {g[0].get('runtime')} (record {g[0]['tag']}, {g[0]['log']})")
        mask = lambda xs: tuple(re.sub(r"ptr=0x[0-9a-f]+|^[IWEFV]\d{4} \d\d:\d\d:\d\d\.\d+ +\d+ ", "…", x) for x in (xs or []))
        for v in VARIANTS:
            rows = [r for r in g if r["variant"] == v]
            sigs = {(mask(r["engine"].get("registered")), mask(r["engine"].get("delegate"))) for r in rows}
            cl = sorted({x for r in rows for x in r["engine"].get("delegate") or [] if "LITERT_CL" in x})
            if rows:
                p(f"- {v}: {len(rows)} GPU records, {len(sigs)} distinct registered + delegate line set(s) (pointers and log "
                  f"stamps masked); LITERT_CL lines: " + " | ".join(x.split(" node(s)")[0] + " " + x.split("subgraph ")[-1] for x in cl))
    status = {}
    for r in recs:
        status[r["status"]] = status.get(r["status"], 0) + 1
    p(f"\n## statuses: {status}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
