# litert-fidelity-runner

Mac runner for the backend-fidelity suite (`methodology/fidelity.md`). The executable is
still named `litert-mac-verify`: it is the yardstick source in `tools/litert-mac-verify/`
plus the `--suite` batch mode. It lives in its own directory because that yardstick is
pinned to the exact wrapper revision behind the published GSM8K rows and must not move.

```bash
./setup.sh    # clones swift-litert-lm @ 84b5df7, applies wrapper.patch, builds
.build/release/litert-mac-verify <model.litertlm> \
  --suite ../../evaldata/fidelity_suite_v1.jsonl --out out.jsonl \
  --backend cpu|gpu --tier smoke|core|full --reps N --runtime-version v0.17.1 \
  [--suite-max-num-tokens M] [--activation f32] [--resume] [--timeout S]
```

- The LiteRT-LM version is whatever `wrapper.patch` pins (v0.17.1). `--runtime-version` only
  labels the records; change the pins and the label together.
- `LITERTLM_VERSION=v0.18.0 ./setup.sh` builds against v0.18.0 instead: it resets the clone to
  84b5df7 and applies `wrapper-v0.18.0.patch` (the same override, xcframework pins moved to the
  checksums in LiteRT-LM's `Package.swift` at v0.18.0). The build replaces `.build/release`;
  plain `./setup.sh` switches back to v0.17.1. Label the records `--runtime-version v0.18.0`.
- `--activation f32` and `--suite-max-num-tokens M` change the recorded backend label
  (`gpu-f32`, `gpu-ctx8192`), so the scorer treats them as separate arms.
- `maxNumTokens` is total context for this runtime. Exit code 3 means one generation timed
  out; rerun with `--resume`.
- `wrapper.patch` adds one non-upstream global to the vendored Swift wrapper
  (`liteRTLMActivationDataTypeOverride`), which calls
  `litert_lm_engine_settings_set_activation_data_type`.
