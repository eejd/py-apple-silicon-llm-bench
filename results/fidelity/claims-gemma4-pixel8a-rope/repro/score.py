#!/usr/bin/env python3
"""score.py — LiteRT-LM #3913 on a Pixel 8a: one device leg of `litert_lm_advanced_main` (one prompt, one process) ->
one record in the form of PowerBeef's #3913 runner (research/issue-3913/runner/Sources/runner/main.swift), stdlib only.

    python3 -I score.py leg <round dir> <tag>     # reads <tag>.out / .err / .meta.json (llm_prompt_chain.sh), appends
                                                  # one line to <raw dir>/<round>__E2B-<variant>__<be>-<prec>.jsonl
                                                  # (a record of the same leg already there: nothing written)
                                                  # exit 3 = the prefill differs from the Mac's for the same question
                                                  # (the chain stops the round), else 0
    python3 -I score.py done <tag>                # exit 0 when a record of that leg tag exists (the chain's skip test)
    python3 -I score.py tags                      # every recorded leg tag, one per line (lane_done_tags)
    python3 -I score.py smoke <research/issue-3913 dir>   # his Mac records -> fake CLI stdout / stderr -> this parser;
                                                  # exact / firstMismatch / firstMismatchPosition must equal his values
    Environment: G3913_RAW_DIR = the raw dir (default out/raw here; lane.env passes RESULTS_DIR), G3913_MODELS = the
    models.tsv (default models.tsv here), G3913_MAC_RAW = the Mac run's raw dir for the prefill check (default
    ../../claims-gemma4-v0.18.0-mac-rope/raw), G3913_UPSTREAM = PowerBeef's research/issue-3913 dir (for `smoke`).

The reply: the CLI's stdout minus log lines (`INFO: Loaded OpenCL library with dlopen.` lands there on the GPU
path) minus channel blocks (`[<channel>] ...[/<channel>]`, PrintMessage's non-streaming form) minus a leading
`output: ` (the Session path prints one; the Conversation path, used here, does not: litert_lm_lib.cc v0.18.0 line 366
vs 1230). Scoring = the runner's: digits = every character with a Unicode numeric type (Swift `Character.isNumber`,
Python `str.isnumeric`), exact = digits == expected, firstMismatch = the first index where they differ, else the
shorter length; firstMismatchPosition = prefill - 1 + firstMismatch. prefill = the prompt's token count = the sum of the
BenchmarkInfo block's prefill turns (the runner records `lastPrefillTokenCount`, the C API's count at index
num_turns - 1, which is the same number when the conversation prefills in one turn, as on his Mac: 1705 for s104; the
Conversation path can split a prompt into turns (seen with Qwen3 on the same phone: 1338 + 2) and the reply's positions
follow the sum); the
turns are kept in `prefill_turns`, the last one in `prefill_last_turn`; decode = the last decode turn.
seconds = the last prefill turn's + the last decode turn's duration (the runner
times sendMessage; the CLI prints no such wall: labelled `seconds_src`). wall_s = the process wall on the phone
(engine load, GPU program compile or cache load, prefill, decode).
"""
import glob
import json
import os
import re
import sys

KIT = os.path.dirname(os.path.abspath(__file__))      # .../claims-gemma4-pixel8a-rope/repro
BASE = os.path.dirname(KIT)                           # .../claims-gemma4-pixel8a-rope (raw/, logs/)
FID = os.path.dirname(BASE)                           # .../results/fidelity
RAW = os.environ.get("G3913_RAW_DIR") or os.path.join(KIT, "out", "raw")
MANIFEST = os.path.join(KIT, "prompts", "manifest.json")
MODELS = os.environ.get("G3913_MODELS") or os.path.join(KIT, "models.tsv")
MAC_RAW = os.environ.get("G3913_MAC_RAW") or os.path.join(FID, "claims-gemma4-v0.18.0-mac-rope", "raw")
UP = os.environ.get("G3913_UPSTREAM", "")

RE_LOG = re.compile(r"^(INFO|WARNING|ERROR|VERBOSE|DEBUG|FATAL): |^[IWEFV]\d{4} \d\d:\d\d:\d\d")
RE_CHANNEL = re.compile(r"(?s)\[([A-Za-z_][\w-]*)\] .*?\[/\1\]\n?")
RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE_PREFILL = re.compile(r"Prefill Turn (\d+): Processed (\d+) tokens in (\S+) duration")
RE_DECODE = re.compile(r"Decode Turn (\d+): Processed (\d+) tokens in (\S+) duration")
RE_DUR = re.compile(r"([0-9.]+)(h|ms|us|µs|ns|m|s)")
RE_PSPEED = re.compile(r"Prefill Speed: ([0-9.]+) tokens/sec")
RE_DSPEED = re.compile(r"Decode Speed: ([0-9.]+) tokens/sec")
RE_TTFT = re.compile(r"Time to first token: ([0-9.]+) s")
RE_INIT = re.compile(r"Init Total: ([0-9.]+) ms")
UNIT = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}


def seconds(text):
    """absl::FormatDuration text (e.g. 2.300594809s, 299.351277ms, 1m2.5s) -> seconds."""
    parts = RE_DUR.findall(text)
    return sum(float(v) * UNIT[u] for v, u in parts) if parts and "".join(v + u for v, u in parts) == text else None


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def reply_text(out):
    out = RE_ANSI.sub("", out)
    body = "\n".join(l for l in out.splitlines() if not RE_LOG.match(l))
    channels = RE_CHANNEL.findall(body)
    body = RE_CHANNEL.sub("", body)
    m = re.search(r"(?s)(?:^|\n)output: (.*)", body)
    if m:
        body = m.group(1)
    return body.strip("\n"), channels


def first_block(err):
    """The text of the first BenchmarkInfo block (one process, one iteration: there is one)."""
    i = err.find("BenchmarkInfo:")
    if i < 0:
        return ""
    j = err.find("Aggregated BenchmarkInfo", i + 1)
    k = err.find("BenchmarkInfo:", i + 1)
    ends = [x for x in (j, k) if x > 0]
    return err[i:min(ends)] if ends else err[i:]


def bench(err):
    b = first_block(err)
    pre = [(int(n), int(t), seconds(d)) for n, t, d in RE_PREFILL.findall(b)]
    dec = [(int(n), int(t), seconds(d)) for n, t, d in RE_DECODE.findall(b)]
    ps, ds = RE_PSPEED.findall(b), RE_DSPEED.findall(b)
    ttft, init = RE_TTFT.search(b), RE_INIT.search(b)
    return {"block": bool(b), "prefill_turns": [t for _, t, _ in pre], "decode_turns": [t for _, t, _ in dec],
            "prefill": sum(t for _, t, _ in pre) if pre else None, "prefill_last_turn": pre[-1][1] if pre else None,
            "decode": dec[-1][1] if dec else None,
            "prefill_s": sum(s for _, _, s in pre) if pre and None not in [s for _, _, s in pre] else None,
            "decode_s": dec[-1][2] if dec else None,
            "prefillTPS": float(ps[-1]) if ps else None, "decodeTPS": float(ds[-1]) if ds else None,
            "ttft_s": float(ttft.group(1)) if ttft else None, "init_total_ms": float(init.group(1)) if init else None}


def compare(reply, expected):
    """The runner's scoring, verbatim in effect."""
    got = [c for c in reply if c.isnumeric()]
    want = list(expected)
    if got == want:
        return True, None
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            return False, i
    return False, min(len(got), len(want))


def engine_lines(err):
    """The GPU's identity, verbatim from the log: the first registered line, the distinct delegate replacement lines,
    the activation line, the OpenCL / WebGPU device lines."""
    lines = err.splitlines()
    pick = lambda rx, n=1: [l.strip() for l in lines if re.search(rx, l)][:n]
    return {"registered": pick(r"Dynamically loaded GPU accelerator\(.*\) registered|RegisterAccelerator: ptr=\S+, name=", 3),
            "delegate": sorted(set(re.sub(r"^.*?(Replacing)", r"\1", l.strip()) for l in lines if "Replacing" in l and "delegate" in l))[:8],
            "activation": pick(r"activation type for Text decoder is|activation_data_type: ", 2),
            "device": pick(r"Created OpenCL device|Selected adapter|delegate_webgpu|delegate_opencl\.cc", 3),
            "weights": pick(r"Weights preparation on Gpu", 1)}


def precision(meta, err, f32):
    """The activation precision of a leg and where it comes from. The Gemma 4 E2B logs print only
    `activation_data_type: Not set` (the CLI's settings before the engine): LiteRT-LM then takes the bundle's
    `prefer_activation_type` of tf_lite_prefill_decode (b7c40ea81 runtime/engine/engine_settings.cc 114-137, 535; the
    same rule in v0.18.0), the GPU compilation options turn it into GpuOptions kFp16 / kFp32 (runtime/executor/
    llm_executor_settings_utils.cc 91-95) and the CPU compilation options take no activation type (275-301)."""
    if f32:
        return "fp32", "--force_f32=true"
    m = re.search(r"System's default activation type for Text decoder is (\w+)", err)
    if m:
        return m.group(1), "log: System's default activation type for Text decoder"
    if meta.get("backend") == "cpu":
        return "fp32", "CPU: the compilation options take no activation type (llm_executor_settings_utils.cc 275-301)"
    peek = read(os.path.join(KIT, "sampler", meta.get("label", "") + ".peek.txt"))
    p = re.search(r"tf_lite_prefill_decode\s*\n\s*Key: prefer_activation_type, Value \(String\): (\w+)", peek)
    if p:
        return p.group(1), ("GPU: activation_data_type not set -> the bundle's tf_lite_prefill_decode prefer_activation_type "
                            f"{p.group(1)} (litert-lm-peek, repro/sampler/{meta.get('label')}.peek.txt) -> GpuOptions precision")
    return "default", "no log line and no peek of the bundle"


def load_models():
    m = {}
    for line in read(MODELS).splitlines():
        if line.strip() and not line.startswith("#"):
            label, variant, path, sha, size = line.split()[:5]
            m[label] = {"variant": variant, "path": path, "sha256": sha, "bytes": int(size)}
    return m


def mac_prefill(cid):
    """The Mac's prefill for the same question: the Mac run's E2B original fp16 records (G3913_MAC_RAW) when they exist
    (same id, sentences, late and seed = the same prompt text), else PowerBeef's E4B Mac records (same Gemma 4 tokenizer
    and template; picitujeromanov's E2B positions agree)."""
    case = json.loads(read(MANIFEST) or "{}").get(cid, {})
    for f in sorted(glob.glob(os.path.join(MAC_RAW, "*E2B-original*gpu-fp16*.jsonl"))):
        for line in read(f).splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if (d.get("id") == cid and d.get("prefill") and d.get("sentences") == case.get("sentences")
                    and d.get("late") == case.get("late") and d.get("seed") == case.get("seed")):
                return int(d["prefill"]), os.path.relpath(f, FID)
    man = json.loads(read(MANIFEST) or "{}").get(cid, {})
    v = man.get("his_mac_e4b_prefill") or []
    if len(v) == 1:
        return int(v[0]), "PowerBeef results/mac (E4B, same sentences/late)"
    return None, None


def leg_record(rdir, tag):
    meta = json.loads(read(os.path.join(rdir, tag + ".meta.json")) or "{}")
    out, err = read(os.path.join(rdir, tag + ".out")), read(os.path.join(rdir, tag + ".err"))
    man = json.loads(read(MANIFEST))
    cid = meta["prompt_id"]
    case = man[cid]
    models = load_models()
    mdl = models.get(meta["label"], {})
    b = bench(err)
    reply, channels = reply_text(out)
    f32 = meta.get("f32", False)
    prec, prec_src = precision(meta, err, f32)
    status = meta.get("status", "?")
    if status == "DONE" and not b["block"]:
        status = "NO_BENCHMARK_INFO"
    rec = {"id": cid, "backend": meta["backend"], "activation": "f32" if f32 else "default",
           "model": os.path.basename(mdl.get("path", meta["label"])), "sentences": case["sentences"], "late": case["late"],
           "seed": case["seed"], "prefill": b["prefill"], "decode": b["decode"], "expected": case["expected"],
           "output": reply,
           "seconds": round(b["prefill_s"] + b["decode_s"], 2) if b["prefill_s"] is not None and b["decode_s"] is not None else None,
           "seconds_src": "BenchmarkInfo prefill turns + last decode turn duration",
           "prefillTPS": b["prefillTPS"], "decodeTPS": b["decodeTPS"],
           "wall_s": meta.get("wall_s"), "cold": meta.get("cache") == "cold", "device": "pixel8a",
           "runtime": meta.get("runtime"), "status": status, "exit": meta.get("rc"),
           "prefill_turns": b["prefill_turns"], "prefill_last_turn": b["prefill_last_turn"],
           "decode_turns": b["decode_turns"], "ttft_s": b["ttft_s"],
           "init_total_ms": b["init_total_ms"], "channels_stripped": channels,
           "variant": mdl.get("variant"), "model_sha256": mdl.get("sha256"), "precision": prec,
           "precision_src": prec_src,
           "tag": tag, "round": meta.get("round"), "ticket": meta.get("ticket"), "cmd": meta.get("cmd"),
           "engine": engine_lines(err), "log": os.path.relpath(os.path.join(rdir, tag), BASE)}
    exact = None
    if status == "DONE" and b["prefill"] is not None:
        exact, mm = compare(reply, case["expected"])
        rec["exact"] = exact
        if not exact:
            rec["firstMismatch"] = mm
            rec["firstMismatchPosition"] = b["prefill"] - 1 + mm
    else:
        rec["exact"] = False
    ref, src = mac_prefill(cid)
    rec["prefill_mac"], rec["prefill_mac_src"] = ref, src
    rec["prefill_match"] = (b["prefill"] == ref) if (b["prefill"] is not None and ref is not None) else None
    return rec, prec


def raw_file(rec, rnd, prec):
    return os.path.join(RAW, f"{rnd}__E2B-{rec['variant']}__{rec['backend']}-{prec}.jsonl")


def recorded_tags():
    tags = set()
    for f in glob.glob(os.path.join(RAW, "*.jsonl")):
        for line in read(f).splitlines():
            try:
                tags.add(json.loads(line).get("tag"))
            except ValueError:
                pass
    return tags


def cmd_leg(rdir, tag):
    rec, prec = leg_record(rdir, tag)
    os.makedirs(RAW, exist_ok=True)
    if tag in recorded_tags():
        print(f"{tag}: already recorded, nothing written")
        return 0
    path = raw_file(rec, rec["round"] or "r2", prec)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{tag}: status {rec['status']} exit {rec['exit']} prefill {rec['prefill']} (turns {rec['prefill_turns']}, mac "
          f"{rec['prefill_mac']} {'MATCH' if rec['prefill_match'] else 'DIFF' if rec['prefill_match'] is False else '?'}) "
          f"decode {rec['decode']} exact {rec['exact']} firstMismatchPosition {rec.get('firstMismatchPosition')} "
          f"wall {rec['wall_s']} s -> {os.path.relpath(path, BASE)}")
    if rec["status"] == "DONE" and rec["prefill_match"] is False:
        print(f"STOP: prefill {rec['prefill']} != the Mac's {rec['prefill_mac']} ({rec['prefill_mac_src']}) for {rec['id']}")
        return 3
    return 0


def cmd_done(tag):
    return 0 if tag in recorded_tags() else 1


def cmd_smoke(up):
    """His Mac records as the CLI would print them: stdout = an OpenCL log line + the reply + newline (PrintMessage);
    stderr = a BenchmarkInfo block with his prefill / decode counts. The parser must give back his exact /
    firstMismatch / firstMismatchPosition for every record, and the channel / Session forms must parse the same."""
    files = sorted(glob.glob(os.path.join(up, "results", "mac", "*.jsonl"))) if up else []
    if not files:
        print(f"smoke: no results/mac/*.jsonl under {up!r}: pass PowerBeef's research/issue-3913 directory")
        return 2
    assert abs(seconds("1m2.5s") - 62.5) < 1e-9 and abs(seconds("299.351277ms") - 0.299351277) < 1e-12
    assert seconds("2.300594809s") == 2.300594809 and seconds("12x") is None
    n = bad = 0
    for f in files:
        for line in read(f).splitlines():
            d = json.loads(line)
            if "expected" not in d or "prefill" not in d:
                continue
            out = "INFO: Loaded OpenCL library with dlopen.\n" + d["output"] + "\n"
            err = ("I0000 00:00:1.0 1 litert_lm_lib.cc:431] BenchmarkInfo:\n  Init Phases (1):\n    - Init Total: 1.00 ms\n"
                   "  Time to first token: 1.00 s\n  Prefill Turns (Total 1 turns):\n"
                   f"    Prefill Turn 1: Processed {d['prefill']} tokens in 1.5s duration.\n      Prefill Speed: 1.00 tokens/sec.\n"
                   "  Decode Turns (Total 1 turns):\n"
                   f"    Decode Turn 1: Processed {d.get('decode', 49)} tokens in 250.5ms duration.\n      Decode Speed: 1.00 tokens/sec.\n")
            variants = {"plain": out, "channel": "[thought] the 7 dwarves 123[/thought]\n" + out,
                        "session": "output: " + d["output"] + "\n", "ansi": "\x1b[33m" + d["output"] + "\x1b[0m\n"}
            # the same prompt prefilled in two turns (N - 2, then 2: the Conversation path's split, seen in a Qwen3 log)
            err2 = err.replace(f"    Prefill Turn 1: Processed {d['prefill']} tokens in 1.5s duration.\n",
                               f"    Prefill Turn 1: Processed {d['prefill'] - 2} tokens in 1.5s duration.\n"
                               "    Prefill Turn 2: Processed 2 tokens in 0s duration.\n")
            assert err2 != err
            for form, o, e in [(f, o, err) for f, o in variants.items()] + [("two-turn", out, err2)]:
                reply, _ = reply_text(o)
                b = bench(e)
                exact, mm = compare(reply, d["expected"])
                pos = b["prefill"] - 1 + mm if mm is not None else None
                ok = (reply == d["output"] and exact == d["exact"] and mm == d.get("firstMismatch")
                      and pos == d.get("firstMismatchPosition") and b["prefill"] == d["prefill"]
                      and abs(b["prefill_s"] + b["decode_s"] - 1.7505) < 1e-9)
                n += 1
                if not ok:
                    bad += 1
                    if bad <= 5:
                        print(f"MISMATCH {os.path.basename(f)} {d['id']} form {form}: exact {exact}/{d['exact']} mm {mm}/"
                              f"{d.get('firstMismatch')} pos {pos}/{d.get('firstMismatchPosition')} reply {reply[:60]!r}")
    print(f"smoke: {n - bad} / {n} parses agree with his records ({len(files)} files x 5 forms: stdout plain, channel, "
          f"session, ansi; a two-turn prefill)")
    return 1 if bad else 0


def main(argv):
    if len(argv) >= 4 and argv[1] == "leg":
        return cmd_leg(argv[2], argv[3])
    if len(argv) == 3 and argv[1] == "done":
        return cmd_done(argv[2])
    if len(argv) == 2 and argv[1] == "tags":
        print("\n".join(sorted(t for t in recorded_tags() if t)))
        return 0
    if len(argv) in (2, 3) and argv[1] == "smoke":
        return cmd_smoke(argv[2] if len(argv) == 3 else UP)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
