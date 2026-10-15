#!/usr/bin/env python3
"""standin_cli.py — a stand-in for `litert_lm_advanced_main` on devicebench's fake phone (dryrun_llm.sh
stages it under the binary's name; it never runs on a phone). It answers one prompt like the CLI's Conversation path:
stdout = an OpenCL log line (GPU) + the reply + newline; stderr = the engine's log lines in the CLI's own forms (the
registered accelerator, the delegate, the activation default, one BenchmarkInfo block), copied from an earlier Pixel 8a log of the same CLI.

The reply copies the prompt's REFERENCE digits; on the GPU with the original model (label `orig`) every digit whose
position (prefill - 1 + k) is above 2047 is doubled, a stand-in for the #3913 symptom, so the scorer sees both exact and
inexact rows. The prefill count = the prompt's `his_mac_e4b_prefill` from prompts/manifest.json (the Mac's count), or,
with FAKE_PREFILL_SHIFT, that count plus the shift (a template mismatch for the scorer's stop). Lives FAKE_LEG_S seconds
(default 3) with $FAKE_ROOT/_run/<name>.pid and $FAKE_ROOT/proc/<pid>/{cmdline,status} for the chain's pidof and sampler;
the first process of prompt FAKE_SLOW_PROMPT sleeps FAKE_SLOW_S (default 200) more (a process the clock must stop);
with FAKE_FAIL_LABEL = a label (or `all`) the process fails at once (exit 1, an error line, no BenchmarkInfo).
A cold GPU leg (no <model>_*.bin cache beside the model) writes one cache file.
"""
import json
import os
import re
import sys
import time

args = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "true") for a in sys.argv[1:] if a.startswith("--"))
root = os.environ.get("FAKE_ROOT", "")
name = os.path.basename(sys.argv[0])
pid = os.getpid()
if root:
    os.makedirs(os.path.join(root, "_run"), exist_ok=True)
    with open(os.path.join(root, "_run", name + ".pid"), "w") as fh:
        fh.write(f"{pid}\n")
    pdir = os.path.join(root, "proc", str(pid))
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "cmdline"), "w") as fh:
        fh.write("\0".join(sys.argv) + "\0")
    with open(os.path.join(pdir, "status"), "w") as fh:
        fh.write("VmHWM:\t 3456789 kB\nVmRSS:\t 3123456 kB\nVmSwap:\t       0 kB\n")
    with open(os.path.join(root, "_run", "leg_started"), "a") as fh:
        fh.write(f"{time.time():.3f} {args.get('input_prompt_file', '')}\n")
try:
    backend = args.get("backend", "cpu")
    model = args.get("model_path", "")
    prompt_path = args.get("input_prompt_file", "")
    prompt = open(prompt_path, encoding="utf-8").read()
    digits = re.search(r"REFERENCE: (\d+)", prompt).group(1)
    man = json.load(open(os.path.join(os.path.dirname(prompt_path), "manifest.json")))
    pid_ = os.path.basename(prompt_path)[:-4]
    pre = (man.get(pid_, {}).get("his_mac_e4b_prefill") or [len(prompt) // 5])[0] + int(os.environ.get("FAKE_PREFILL_SHIFT", "0"))
    label = os.path.basename(model).split(".")[0]
    if os.environ.get("FAKE_FAIL_LABEL") in (label, "all"):
        sys.stderr.write(f"ERROR: [fake] {label}: failed to create the engine (FAKE_FAIL_LABEL)\n")
        sys.exit(1)
    reply ="".join(d * (2 if (backend == "gpu" and label == "orig" and pre - 1 + k > 2047) else 1) for k, d in enumerate(digits))
    err = sys.stderr
    err.write("WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n")
    err.write(f"I0000 00:00:1.0 1 litert_lm_lib.cc:654] Choose backend: {backend}\n")
    err.write("activation_data_type: Not set\n")
    err.write("W0000 00:00:1.0 1 litert_lm_loader.cc:539] TFLite model type: tf_lite_vision_encoder not found for prefer "
              "activation type. Use system's default backend activation type. System's default activation type for Text "
              f"decoder is {'fp16' if backend == 'gpu' and args.get('force_f32') != 'true' else 'fp32'}. Vision encoder is fp32.\n")
    if backend == "gpu":
        err.write("INFO: [gpu_registry.cc:130] Attempting to load GPU accelerator(libLiteRtGpuAccelerator.so).\n")
        err.write("INFO: [accelerator_registry.cc:54] RegisterAccelerator: ptr=0x1, name=LiteRT GPU\n")
        err.write("INFO: [gpu_registry.cc:138] Dynamically loaded GPU accelerator(libLiteRtGpuAccelerator.so) registered.\n")
        err.write("INFO: [gpu_environment.cc:239] Created OpenCL device from provided device id and platform id.\n")
        err.write("VERBOSE: Replacing 436 out of 436 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 0 (prefill_128).\n")
        err.write("W0000 00:00:1.0 1 gpu_model_builder.cc:4452] Weights preparation on Gpu is disabled for PowerVR, Broadcom, Mali GPUs and non-OpenCL/WebGPU/Metal backends.\n")
        cache = model + "_fake_gpu_cache.bin"
        if not os.path.exists(cache):
            with open(cache, "wb") as fh:
                fh.write(b"\0" * 4096)
    else:
        err.write("INFO: [cpu_registry.cc:73] XNNPACK CPU accelerator registered.\n")
    err.write("I0000 00:00:1.0 1 litert_lm_lib.cc:1119] Creating conversation\n")
    err.write("I0000 00:00:1.0 1 litert_lm_lib.cc:1131] Running single-turn conversation\n")
    slow = os.environ.get("FAKE_SLOW_PROMPT", "")
    marker = os.path.join(root, "_run", "slow_done") if root else ""
    if slow and pid_ == slow and marker and not os.path.exists(marker):
        open(marker, "w").write(f"{time.time():.3f}\n")          # slow once: the chain's clock must stop this one
        time.sleep(float(os.environ.get("FAKE_SLOW_S", "200")))
    time.sleep(float(os.environ.get("FAKE_LEG_S", "3")))
    dec = len(reply) + 1
    err.write("I0000 00:00:2.0 1 litert_lm_lib.cc:431] BenchmarkInfo:\n  Init Phases (2):\n    - Init Executor: 1500.00 ms\n"
              "    - Init Total: 1600.00 ms\n--------------------------------------------------\n  Time to first token: 2.10 s\n"
              "--------------------------------------------------\n  Prefill Turns (Total 1 turns):\n"
              f"    Prefill Turn 1: Processed {pre} tokens in 2.047s duration.\n      Prefill Speed: {pre / 2.047:.2f} tokens/sec.\n"
              "--------------------------------------------------\n  Decode Turns (Total 1 turns):\n"
              f"    Decode Turn 1: Processed {dec} tokens in 1m0.5s duration.\n      Decode Speed: {dec / 60.5:.2f} tokens/sec.\n"
              "--------------------------------------------------\n")
    if backend == "gpu":
        sys.stdout.write("INFO: Loaded OpenCL library with dlopen.\n")
    sys.stdout.write(reply + "\n")
    sys.stdout.flush()
finally:
    if root:
        try:
            os.remove(os.path.join(root, "_run", name + ".pid"))
        except OSError:
            pass
