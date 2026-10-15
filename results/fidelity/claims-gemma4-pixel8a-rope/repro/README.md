# Repro kit: Gemma 4 E2B on a Pixel 8a through LiteRT-LM's Android CLI (LiteRT-LM #3913)

What produced `../raw/` and `../logs/` (`claims-gemma4-pixel8a-rope.md`), in the form it ran: the
prompts, the plan of the legs, the settings of the device chain, the scorer and the recount. Python 3
standard library and bash only. The chain itself is `community_accel_work/devicebench/llm_prompt_chain.sh`
in [litertlm-convert](https://github.com/john-rocky/litertlm-convert) (README section "LLM prompt legs").

| file | what |
|---|---|
| `gen_prompts.py` | PowerBeef's runner's prompt generator (`research/issue-3913/runner`, Swift) ported to Python: the LCG reference digits, the filler text, the instruction line. `python3 -I gen_prompts.py <his research/issue-3913 dir>` writes the 67 prompt files into `prompts/` and compares them with `prompts/SHA256SUMS` and `prompts/manifest.json`; with his `results/` present it also checks `expected` against every record of his published results. |
| `prompts/SHA256SUMS`, `prompts/manifest.json` | the 67 prompts' sha256 (the chain checks them on the phone) and, per case: cases file, sentences, late, seed, expected digits, bytes, subset (`gpu26` = the 26 run with all three files, `cpu3` = the CPU controls), his E4B Mac prefill for the same prompt |
| `plan_r2.tsv`, `gen_plan.py` | the 129 legs in the order they were run (the generator writes the file from the manifest); the first 96 ran inside the reservation |
| `lane.env` | the chain's settings: the serial, the stage directory on the phone, the clock, `--disable_input_prompt_as_hint=true`, the scorer hooks |
| `models.tsv` | the three model files with sha256 and byte count; put their directory in place of `MODELS_DIR` |
| `kit.SHA256SUMS` | `litert_lm_advanced_main` and the seven `.so` of LiteRT-LM `main` at b7c40ea81 (android_arm64); `LLM_KIT` names a directory holding exactly these eight files and this list as `SHA256SUMS` |
| `score.py` | one leg's stdout, stderr and `meta.json` → one record in the form of PowerBeef's runner: digits = every character `str.isnumeric`, `exact`, `firstMismatch`, `firstMismatchPosition` = prefill − 1 + firstMismatch, the prefill from the BenchmarkInfo block, the engine's identity lines. `score.py smoke <his dir>` replays his Mac records through the parser in five output forms. |
| `sampler/*.peek.txt` | `litert-lm-peek` of the three bundles: no `sampler_params` (so the runtime's default greedy sampler), `prefer_activation_type` fp16 for the text decoder (the GPU precision written into the records) |
| `count.py` | the tables of `../count.txt` from `../raw/*.jsonl`: `python3 -I count.py > ../count.txt` |
| `standin_cli.py`, `plan_dry.tsv` | the fake CLI and the plan of devicebench's dry run (`bash dryrun_llm.sh lane.env`): six scenes on a fake phone, no device |

## Running it again

1. Prompts: `python3 -I gen_prompts.py <his research/issue-3913 dir>` (PowerBeef/LiteRT-LM, branch
   `research/3913` at 39d5a014) → `67 / 67 files unchanged`.
2. Models: the three files of `claims-gemma4-v0.18.0-mac-rope.md` (the original from litert-community,
   the two patched files from his `patch_rope.py --split`, block 2048 and block 512); edit the path
   column of `models.tsv`.
3. The kit: `litert_lm_advanced_main` built for android_arm64 from LiteRT-LM `main` at b7c40ea81 with
   that commit's prebuilt `.so`; the sha256 must equal `kit.SHA256SUMS` (the chain refuses otherwise).
4. `bash llm_prompt_chain.sh check r2a lane.env`, then `run r2a lane.env` with `LLM_KIT` set. Records
   go to `out/raw/`, per-leg logs to `out/logs/r2a/`.
5. `python3 -I count.py out/raw > count.txt`.

## Path fields in the records

`log` and `prefill_mac_src` are paths of the run's working directory: `logs/android/r2a/<tag>` is
`../logs/<tag>` here and `raw/mac/<file>` is `../../claims-gemma4-v0.18.0-mac-rope/raw/<file>`;
`precision_src` names `K/android/sampler/<label>.peek.txt`, which is `sampler/<label>.peek.txt` here. A
rerun writes them relative to this kit. `../logs/chain.log` is the chain's log of the reservation: `K/` in it
is that working directory (its `android/` is this kit, its `raw/android/` is `../raw/`), and one queue
line's session name is replaced by a placeholder.

## Checked from this layout on the Mac (2026-10-15)

`gen_prompts.py`: 67 / 67 unchanged, manifest unchanged, 692 / 692 of his records. `gen_plan.py`
reproduces `plan_r2.tsv` and `plan_dry.tsv` byte for byte. `score.py smoke`: 2,695 / 2,695. Four of the
run's legs rescored from their logs equal the published records in every field but the three path texts.
`count.py` reproduces `../count.txt`. The dry run (`dryrun_llm.sh lane.env` from this directory): 33 / 33 checks
in its six scenes, exit 0.
