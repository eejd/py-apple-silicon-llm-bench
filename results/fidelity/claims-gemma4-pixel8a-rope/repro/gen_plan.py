#!/usr/bin/env python3
"""gen_plan.py — the leg plans of llm_prompt_chain.sh from prompts/manifest.json (stdlib only).

    python3 -I gen_plan.py      # writes plan_r2.tsv (the device round) and plan_dry.tsv (the fake phone's dry run)

plan_r2.tsv, in order (one model on the phone at a time; the chain evicts a model once no undone leg needs it):
  1. orig  gpu  the 26 subset questions, s104-seed1 first and cold (the GPU program compile)
  2. orig  cpu  the first control s104-seed1 (cold: the XNNPACK weight cache) while the original is on the phone
  3. r512  gpu  the same 26, first cold
  4. r2048 gpu  the same 26, first cold
  5. orig  cpu  the other two controls s125-seed1 / s129-seed1 (the original pushed again)
  6. orig  gpu  the other 24 questions of `baseline` (late-* included), then r512 gpu the same 24 (if time is left)
(One reservation of the phone holds the first four groups: original 26 + rope512 26 come before anything else that
costs a push.)
Columns: tag label backend prompt max_num_tokens max_output_tokens cache timeout_s est_s rest_s [extras]
(max_num_tokens 8192 = the runner's default, max_output_tokens 80 = the runner's).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
man = json.load(open(os.path.join(HERE, "prompts", "manifest.json")))
base_order = [k for k, v in man.items() if v["cases_file"] == "baseline"]
long_order = [k for k, v in man.items() if v["cases_file"] == "long"]
# the manifest is a dict sorted by key; restore his file order (sentences, then late, then seed)
order_key = lambda k: (man[k]["late"], man[k]["sentences"], man[k]["seed"])
base_order.sort(key=order_key)
long_order.sort(key=order_key)
subset = [k for k in base_order + long_order if "gpu26" in man[k]["subset"]]
cpu3 = [k for k in base_order + long_order if "cpu3" in man[k]["subset"]]
first = "s104-seed1-iphone1705"
subset.remove(first)
subset.insert(0, first)
rest24 = [k for k in base_order if "gpu26" not in man[k]["subset"]]
assert len(subset) == 26 and len(cpu3) == 3 and len(rest24) == 24, (len(subset), len(cpu3), len(rest24))

HEAD = "# tag  label  backend  prompt  max_num_tokens  max_output_tokens  cache  timeout_s  est_s  rest_s  [extras]\n"


def legs(label, be, ids, tmo_cold, est_cold, tmo, est, cold_first=True):
    out = []
    for n, cid in enumerate(ids):
        cold = cold_first and n == 0
        out.append(f"{label}-{be}-{cid}  {label}  {be}  {cid}  8192  80  {'cold' if cold else 'keep'}  "
                   f"{tmo_cold if cold else tmo}  {est_cold if cold else est}  0")
    return out


plan = ["# G #3913 r2 plan (gen_plan.py): Pixel 8a, LiteRT-LM CLI, one prompt per process, the files as they are (greedy:"
        "# no sampler_params, runtime default k 1)"]
plan += ["# 1. original GPU, the 26 subset questions"] + legs("orig", "gpu", subset, 900, 240, 600, 90)
plan += ["# 2. original CPU, the first control (the launch's slot-A CPU leg), while the original is on the phone"]
plan += legs("orig", "cpu", cpu3[:1], 900, 300, 900, 180)
plan += ["# 3. rope512 GPU, the same 26"] + legs("r512", "gpu", subset, 900, 240, 600, 90)
plan += ["# 4. rope2048 GPU, the same 26"] + legs("r2048", "gpu", subset, 900, 240, 600, 90)
plan += ["# 5. original CPU, the other two controls (the original pushed again)"]
plan += legs("orig", "cpu", cpu3[1:], 900, 300, 900, 180)
plan += ["# 6. the other 24 of baseline (if time is left): original, then rope512"]
plan += legs("orig", "gpu", rest24, 900, 240, 600, 90) + legs("r512", "gpu", rest24, 900, 240, 600, 90)
with open(os.path.join(HERE, "plan_r2.tsv"), "w") as fh:
    fh.write(HEAD + "\n".join(plan) + "\n")

dry = ["# G #3913 r2 dry plan (gen_plan.py): the fake phone, the stand-in CLI, two labels, a CPU leg"]
dry += legs("orig", "gpu", [first, "s125-seed1-iphone2028", "s250-seed1"], 120, 20, 120, 15)
dry += legs("orig", "cpu", [first], 120, 20, 120, 15)
dry += legs("r512", "gpu", ["s125-seed1-iphone2028", "s250-seed1"], 120, 20, 120, 15)
with open(os.path.join(HERE, "plan_dry.tsv"), "w") as fh:
    fh.write(HEAD + "\n".join(dry) + "\n")
n = sum(1 for l in plan if not l.startswith("#"))
print(f"plan_r2.tsv: {n} legs (orig gpu 26 + cpu 3, r512 gpu 26, r2048 gpu 26, extension orig 24 + r512 24); "
      f"plan_dry.tsv: {sum(1 for l in dry if not l.startswith('#'))} legs")
