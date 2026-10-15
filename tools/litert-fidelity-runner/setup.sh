#!/bin/bash
# Fetch the Swift wrapper this runner builds against and apply wrapper.patch
# (xcframework pins -> LiteRT-LM v0.17.1, plus the activation-data-type override).
# LITERTLM_VERSION=v0.18.0 applies wrapper-v0.18.0.patch instead (same override,
# xcframework pins -> v0.18.0). Switching versions resets the clone to 84b5df7 first.
set -euo pipefail
cd "$(dirname "$0")"
LITERTLM_VERSION=${LITERTLM_VERSION:-v0.17.1}
case "$LITERTLM_VERSION" in
  v0.17.1) PATCH=wrapper.patch ;;
  v0.18.0) PATCH=wrapper-v0.18.0.patch ;;
  *) echo "LITERTLM_VERSION must be v0.17.1 or v0.18.0" >&2; exit 2 ;;
esac
if [ ! -d swift-litert-lm ]; then
  git clone https://github.com/john-rocky/swift-litert-lm.git
  git -C swift-litert-lm checkout --detach 84b5df7
fi
if ! git -C swift-litert-lm apply --reverse --check "../$PATCH" 2>/dev/null; then
  git -C swift-litert-lm checkout -- .
  git -C swift-litert-lm apply "../$PATCH"
fi
swift build -c release
echo "built ($LITERTLM_VERSION): $(pwd)/.build/release/litert-mac-verify"
