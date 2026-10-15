# Fidelity claims run — Gemma 4 E2B, LiteRT-LM Android CLI, Pixel 8a: the rotary-position patches from LiteRT-LM #3913

> On a Pixel 8a (Tensor G3, Mali-G715 GPU; Android 16, build CP1A.260505.005), LiteRT-LM's
> `litert_lm_advanced_main` (android_arm64) built from `main` at b7c40ea81 (2026-10-13),
> `--backend=gpu` (the OpenCL GPU accelerator, default fp16 activations): the unmodified
> `gemma-4-E2B-it.litertlm` returns the 48-digit reference exactly in all 15 cases whose reply
> digits sit below token position 2048, in 2 of the 9 cases whose reply crosses 2048, and in none
> of the 17 cases whose reply lies entirely at or beyond 2048 (first wrong digit at positions
> 2051–3998). The same file patched with PowerBeef's `patch_rope.py --split` (block 2048) and with
> the block-512 variant returns all 26 cases run with them exactly.

Cases are PowerBeef's `cases/` (`research/3913` at 39d5a014): a subset of 26 run with all three
files (`baseline` s104 / s117 / s120 / s123 / s125 / s127 × seeds 1–3 = 18 cases, prompts 1705–2061
tokens; `long` s129 / s161 / s190 / s250 × seeds 1–2 = 8 cases, 2094–3983 tokens), plus 15 more
`baseline` cases run with the unmodified file only in the time left (s122 / s124 / s126 / s129 /
s142 × seeds 1–3, prompts 1982–2297). Each case asks the model to copy a 48-digit number placed
before a filler text; `exact` means the reply is the number, digit for digit. The prompts were
generated on the Mac from a Python port of his runner's generator
(`claims-gemma4-pixel8a-rope/repro/gen_prompts.py`: the LCG reference digits and the filler;
`expected` checked against all 692 records of his published results; the 67 prompts' sha256 in
`repro/prompts/SHA256SUMS`) and sent through the CLI's conversation path (`--use_session=false`,
the engine applies the chat template), one process per case. Every count below was recounted from
the jsonl with two separate scripts (`repro/count.py` is one of them; its output is
`claims-gemma4-pixel8a-rope/count.txt`).

Model files are the ones of the Mac run (`claims-gemma4-v0.18.0-mac-rope.md`; sha256 read back on
the phone after the push): `gemma-4-E2B-it.litertlm` `181938105e…9a63c`,
`gemma-4-E2B-it-rope512.litertlm` `a963992d90…40b17` (= picitujeromanov's value on #3913),
`gemma-4-E2B-it-rope2048.litertlm` `0b89958c5d…57a4`.

## Exact replies per arm

Grouped by the positions of the 48 reply digits (prefill−1 … prefill+46):

| file | backend | `baseline`, reply below 2048 | `baseline`, reply crossing 2048 | `baseline`, reply at or above 2048 | `long` | total |
|---|---|---|---|---|---|---|
| original | gpu | 15/15 (positions 1704–2044) | 2/9 (2011–2092) | 0/9 (2060–2343) | 0/8 (2093–4029) | 17/41 |
| original, the 26-case subset only | gpu | 12/12 | 0/3 (s125) | 0/3 (s127) | 0/8 | 12/26 |
| rope2048 (`--split`) | gpu | 12/12 | 3/3 | 3/3 | 8/8 | 26/26 |
| rope512 | gpu | 12/12 | 3/3 | 3/3 | 8/8 | 26/26 |
| original | cpu | 1/1 (s104-seed1) | 1/1 (s125-seed1) | – | 1/1 (s129-seed1) | 3/3 |

First wrong digit of the unmodified file's gpu misses: crossing cases 2051–2069, cases at or above
2048 2069–2309, `long` 2100–3998. No wrong digit below position 2051. The two crossing cases the
unmodified file gets right (`s124-seed1`, `s124-seed3`: prefill 2012, digits at 2011–2058) are the
same two case ids the unmodified E2B file gets right on the Mac.

The prefill token count of every case equals the Mac run's count for the same case id (96 of 96
processes, all one prefill turn), so the phone and the Mac saw the same prompt tokens.

## Not claimed

Root cause, comparison with any other runtime, other Android devices (no Adreno device was run),
E4B on the phone, and the sets not run here (`short`, `cross4096`, `xl`). The GPU path is what the
engine log registered: `Dynamically loaded GPU accelerator(libLiteRtGpuAccelerator.so) registered.`,
`RegisterAccelerator: … name=LiteRT GPU`, `Created OpenCL device from provided device id and
platform id.`, `Loaded OpenCL library with dlopen.`, `delegate_opencl.cc`, and every signature
delegated whole (`Replacing 2068 out of 2068 node(s) with delegate (LITERT_CL) node, yielding 1
partitions` for the unmodified file's decode signature; 2105 of 2105 for the patched files). The
activation line in the log is `activation_data_type: Not set`, i.e. the default; the bundle's
text-decoder metadata asks for fp16 (`prefer_activation_type`), which the runtime applies to the
GPU (`runtime/engine/engine_settings.cc` at b7c40ea81). Greedy decoding comes from the runtime's
defaults for a bundle without `sampler_params` (this bundle has none; the default is top-k 1,
`engine_settings.cc` lines 416–442 at b7c40ea81, the same block as v0.18.0's lines 395–421): the
CLI has no sampler flag. Wall time per process (9–20 s warm, 37–73 s for a file's first load) is
not a benchmark.

## Run record

JST, 2026-10-15, one reservation of the phone (09:20:34 – 10:00:45), USB, one
`litert_lm_advanced_main` process per case with a readiness gate (thermal status, GPU cooling)
before each. Order: original gpu 26 → original cpu s104-seed1 → rope512 gpu 26 → rope2048 gpu 26 →
original cpu s125-seed1, s129-seed1 → original gpu, 15 more `baseline` cases. 96 processes, all
exit 0, no timeouts.

```text
cd /data/local/tmp/g3913_rope && LD_LIBRARY_PATH=/data/local/tmp/g3913_rope ./litert_lm_advanced_main \
  --backend=gpu --model_path=/data/local/tmp/g3913_rope/orig.litertlm \
  --input_prompt_file=/data/local/tmp/g3913_rope/prompts/s104-seed1-iphone1705.txt \
  --max_num_tokens=8192 --max_output_tokens=80 --async=false --benchmark \
  --benchmark_prefill_tokens=0 --benchmark_decode_tokens=0 --use_session=false \
  --disable_input_prompt_as_hint=true
```

`--disable_input_prompt_as_hint=true` keeps the CLI from sizing its prefill batches from the prompt
(the Swift runner on the Mac does not pass that hint). Binary and libraries: `litert_lm_advanced_main`
sha256 `ce1f25dc98…5234`, `libLiteRtGpuAccelerator.so` `c9a77bc7ce…8575`,
`libLiteRtOpenClAccelerator.so` `7c747af055…6ced3` (the android_arm64 build of LiteRT-LM `main` at
b7c40ea8178b80919f7f5af79855c922a40bfaf7, LITERT_REF 9b8c7f77; `main` at that commit is 142 commits
past the point where `release/v0.18.0` branched and is not a descendant of the v0.18.0 tag). Raw
records: `claims-gemma4-pixel8a-rope/raw/r2a__E2B-<file>__<backend>-<activations>.jsonl`, one JSON
line per case with `id`, `prefill`, `exact`, `expected`, `output`, `firstMismatchPosition`, the
process's command line and wall time; engine logs (stdout and stderr of every process):
`claims-gemma4-pixel8a-rope/logs/`, with the chain's own log of the reservation as `logs/chain.log`.
The run's inputs and tools are in `claims-gemma4-pixel8a-rope/repro/` (README there): the prompt
generator and the prompts' checksums, the plan of the legs, the chain's settings, the model and
binary checksums, the scorer that turned each process's output into its record, and the recount.
Three record fields hold paths of the run's working directory: `log` (`logs/android/r2a/<tag>` is
`claims-gemma4-pixel8a-rope/logs/<tag>` here), `prefill_mac_src` (`raw/mac/<file>` is
`claims-gemma4-v0.18.0-mac-rope/raw/<file>`) and `precision_src` (`K/android/sampler/<label>.peek.txt`
is `repro/sampler/<label>.peek.txt`).
