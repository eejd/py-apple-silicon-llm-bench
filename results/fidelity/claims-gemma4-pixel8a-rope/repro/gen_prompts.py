#!/usr/bin/env python3
"""gen_prompts.py — the copy-fidelity prompts of PowerBeef's LiteRT-LM #3913 runner, ported to Python.

Port of `research/issue-3913/runner/Sources/runner/main.swift` (PowerBeef/LiteRT-LM research/3913 @ 39d5a014):
  referenceDigits(seed): 64-bit wrapping LCG, state = seed * 6364136223846793005 + 1442695040888963407, then per digit
      state = state * 6364136223846793005 + 1442695040888963407 and digit = (state >> 33) % 10, 48 digits.
  filler(sentences): "<subjects[i % 12]> <actions[(i / 12 + i) % 9]>." joined by one space.
  body = late ? filler + "\\n\\n" + "REFERENCE: " + digits : "REFERENCE: " + digits + "\\n\\n" + filler
  prompt = body + "\\n\\nCopy the REFERENCE number exactly, digit for digit. Reply with only the digits."
The runner sends the prompt as one user message through `conversation.sendMessage` (the engine renders Gemma 4's
template, no system prompt); the device CLI does the same on its Conversation path (`--use_session=false`), which
reads the prompt file byte for byte (`GetInputPrompt`: the whole file) = each file has no trailing newline.

    python3 -I gen_prompts.py <research/issue-3913 dir> [out dir]
        # <research/issue-3913 dir> = PowerBeef's folder (PowerBeef/LiteRT-LM, branch research/3913 at 39d5a014):
        # cases/ is read; results/ (when present) checks `expected` against every record of his results
        # (results/mac/*.jsonl, results/iphone/*.json) and fills `his_mac_e4b_prefill` in the manifest.
        # Writes <out dir>/<id>.txt + SHA256SUMS + manifest.json (default out dir: prompts/ next to this script) and
        # compares them with the SHA256SUMS and manifest.json already there; exit 1 on any difference or mismatch.

Cases: his `baseline.json` (42) and `long.json` (25); the subset run with all three files on the phone is marked in
the manifest (`subset`: gpu26 / cpu3). Stdlib only.
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "prompts")
M64 = (1 << 64) - 1
A, C = 6364136223846793005, 1442695040888963407

SUBJECTS = [
    "The harbour master", "A visiting engineer", "The night ferry", "Every lighthouse keeper",
    "The council archive", "A quiet fishing village", "The coastal railway", "Old stone breakwaters",
    "The weather station", "A crew of volunteers", "The market square", "Seasonal tourists",
]
ACTIONS = [
    "kept careful notes about the tides and the changing colour of the sea",
    "described how fog rolled in from the bay long before sunrise",
    "noticed that the gulls gathered near the pier whenever storms approached",
    "repaired the wooden walkway after a winter of strong easterly winds",
    "spoke about the slow work of rebuilding the eastern quay",
    "watched cargo ships wait outside the channel for the evening tide",
    "planted hardy grasses along the dunes to hold the sand in place",
    "painted the signal tower white again before the summer festival",
    "explained why the old canal locks were opened only at high water",
]
INSTRUCTION = "Copy the REFERENCE number exactly, digit for digit. Reply with only the digits."

# the subset run with all three files: 18 baseline + 8 long on the GPU; 3 CPU controls (original)
SUBSET_GPU = [f"s{s}-seed{k}-iphone{p}" for s, p in ((104, 1705), (117, 1905), (120, 1953), (123, 1998), (125, 2028),
                                                     (127, 2061)) for k in (1, 2, 3)] + \
             [f"s{s}-seed{k}" for s in (129, 161, 190, 250) for k in (1, 2)]
SUBSET_CPU = ["s104-seed1-iphone1705", "s125-seed1-iphone2028", "s129-seed1"]


def reference_digits(seed):
    state = (seed * A + C) & M64
    out = []
    for _ in range(48):
        state = (state * A + C) & M64
        out.append(str((state >> 33) % 10))
    return "".join(out)


def filler(sentences):
    return " ".join(f"{SUBJECTS[i % len(SUBJECTS)]} {ACTIONS[(i // len(SUBJECTS) + i) % len(ACTIONS)]}."
                    for i in range(sentences))


def prompt_for(case):
    digits = reference_digits(case["seed"])
    reference = "REFERENCE: " + digits
    body = filler(case["sentences"]) + "\n\n" + reference if case["late"] else reference + "\n\n" + filler(case["sentences"])
    return body + "\n\n" + INSTRUCTION, digits


def his_records(up):
    """Every record of his results that carries a seed and an expected value: (file, seed, expected, prefill, sentences,
    late)."""
    recs = []
    for f in sorted(glob.glob(os.path.join(up, "results", "mac", "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    recs.append((os.path.basename(f), d))
    for f in sorted(glob.glob(os.path.join(up, "results", "iphone", "*.json"))):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for d in data if isinstance(data, list) else data.get("results", []):
            recs.append((os.path.basename(f), d))
    return [(f, d) for f, d in recs if "seed" in d and "expected" in d]


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def main(argv):
    if len(argv) < 2 or not os.path.isdir(os.path.join(argv[1], "cases")):
        print(__doc__)
        return 2
    up = os.path.abspath(argv[1])
    out = os.path.abspath(argv[2]) if len(argv) > 2 else OUT
    os.makedirs(out, exist_ok=True)
    # what is there already (the committed list and manifest): the new files must reproduce them
    before = {}
    try:
        with open(os.path.join(out, "SHA256SUMS"), encoding="utf-8") as fh:
            before = {name: sha for sha, name in (line.split() for line in fh if line.strip())}
    except OSError:
        pass
    manifest_before = read_json(os.path.join(out, "manifest.json"))

    cases = []
    for name in ("baseline", "long"):
        with open(os.path.join(up, "cases", name + ".json"), encoding="utf-8") as fh:
            for c in json.load(fh):
                cases.append(dict(c, cases_file=name))
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids across baseline / long"
    missing = [i for i in SUBSET_GPU + SUBSET_CPU if i not in ids]
    assert not missing, f"subset ids not in the cases: {missing}"

    manifest, sums = {}, []
    for c in cases:
        text, digits = prompt_for(c)
        data = text.encode("utf-8")
        path = os.path.join(out, c["id"] + ".txt")
        with open(path, "wb") as fh:
            fh.write(data)
        sha = hashlib.sha256(data).hexdigest()
        sums.append(f"{sha}  {c['id']}.txt")
        subset = [s for s, lst in (("gpu26", SUBSET_GPU), ("cpu3", SUBSET_CPU)) if c["id"] in lst]
        manifest[c["id"]] = {"cases_file": c["cases_file"], "sentences": c["sentences"], "late": c["late"],
                             "seed": c["seed"], "expected": digits, "bytes": len(data), "sha256": sha,
                             "subset": subset}
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sums) + "\n")
    print(f"prompts: {len(cases)} files in {out} (baseline {sum(c['cases_file'] == 'baseline' for c in cases)}, "
          f"long {sum(c['cases_file'] == 'long' for c in cases)}); subset gpu {len(SUBSET_GPU)}, cpu {len(SUBSET_CPU)}")

    # expected digits vs every record of his results (the digits depend on the seed only)
    bad = []
    recs = his_records(up) if os.path.isdir(os.path.join(up, "results")) else []
    if recs:
        bad = [(f, d["seed"], d["expected"]) for f, d in recs if reference_digits(int(d["seed"])) != d["expected"]]
        seeds = sorted({int(d["seed"]) for _, d in recs})
        files = sorted({f for f, _ in recs})
        print(f"expected check: {len(recs) - len(bad)} / {len(recs)} records of his results match reference_digits(seed) "
              f"({len(files)} files, seeds {seeds})")
        for f, s, e in bad[:10]:
            print(f"  MISMATCH {f} seed {s}: his {e} ours {reference_digits(int(s))}")
        # his E4B Mac prefill per (sentences, late) = the template's token count for the same prompt (same tokenizer)
        pre = {}
        for f, d in recs:
            if f.endswith(".jsonl") and "prefill" in d and "sentences" in d:
                pre.setdefault((int(d["sentences"]), bool(d.get("late", False))), set()).add(int(d["prefill"]))
        multi = {k: sorted(v) for k, v in pre.items() if len(v) > 1}
        print(f"his Mac prefill per (sentences, late): {len(pre)} keys, keys with more than one value: {multi or 'none'}")
        for cid in SUBSET_GPU:
            m = manifest[cid]
            m["his_mac_e4b_prefill"] = sorted(pre.get((m["sentences"], m["late"]), []))
    else:
        print(f"no results/ under {up}: the expected check against his records skipped, his_mac_e4b_prefill not filled")
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    # the files that were there: every sha256 and the manifest must come back unchanged
    differs = False
    if before:
        now = {f"{c['id']}.txt": manifest[c["id"]]["sha256"] for c in cases}
        same = [n for n, s in now.items() if before.get(n) == s]
        differs = len(same) != len(now) or len(before) != len(now)
        print(f"against the SHA256SUMS that was here: {len(same)} / {len(now)} files unchanged"
              + ("" if not differs else f"; changed or new {sorted(set(now) - set(same))[:5]}, "
                                        f"gone {sorted(set(before) - set(now))[:5]}"))
    if manifest_before is not None:
        same_m = manifest_before == read_json(os.path.join(out, "manifest.json"))
        differs = differs or not same_m
        print(f"against the manifest.json that was here: {'unchanged' if same_m else 'CHANGED'}")
    return 1 if bad or differs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
