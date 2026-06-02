#!/usr/bin/env bash
# Repackage VOICEVOX.AppImage replacing the CUDA EP with a ROCm-built provider.
#
# Usage:
#   ./repackage-appimage.sh \
#       --rocm-so /path/to/libonnxruntime_providers_rocm.so \
#       --appimage /path/to/VOICEVOX.AppImage \
#       --output   /path/to/VOICEVOX-ROCm.AppImage
#
# Requirements: unsquashfs, mksquashfs (squashfs-tools >= 4.4)

set -euo pipefail

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//' | tail -n +2
  exit 1
}

ROCM_SO=""
INPUT_APPIMAGE=""
OUTPUT_APPIMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm-so)   ROCM_SO="$2";         shift 2 ;;
    --appimage)  INPUT_APPIMAGE="$2";  shift 2 ;;
    --output)    OUTPUT_APPIMAGE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "$ROCM_SO" || -z "$INPUT_APPIMAGE" || -z "$OUTPUT_APPIMAGE" ]] && usage

# --- sanity checks -----------------------------------------------------------
for tool in unsquashfs mksquashfs dd; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not found"; exit 1; }
done

[[ -f "$ROCM_SO" ]]         || { echo "ERROR: ROCm .so not found: $ROCM_SO"; exit 1; }
[[ -f "$INPUT_APPIMAGE" ]]  || { echo "ERROR: AppImage not found: $INPUT_APPIMAGE"; exit 1; }

RUNTIME_SIZE=$("$INPUT_APPIMAGE" --appimage-offset 2>/dev/null) || {
  echo "ERROR: Could not determine AppImage squashfs offset"
  exit 1
}
echo "Runtime size: ${RUNTIME_SIZE} bytes"

# --- work directory ----------------------------------------------------------
WORK=$(mktemp -d --tmpdir voicevox-rocm-repack.XXXXXX)
trap 'echo "Cleaning up $WORK ..."; rm -rf "$WORK"' EXIT

APPDIR="$WORK/appdir"
NEW_SFS="$WORK/new.squashfs"
RUNTIME_BIN="$WORK/runtime.bin"

# --- 1. extract squashfs -----------------------------------------------------
echo "Extracting squashfs (this may take a few minutes) ..."
unsquashfs -o "$RUNTIME_SIZE" -d "$APPDIR" "$INPUT_APPIMAGE"

TARGET="$APPDIR/vv-engine/libvoicevox_onnxruntime_providers_cuda.so"
[[ -f "$TARGET" ]] || { echo "ERROR: Expected $TARGET inside AppImage"; exit 1; }

# --- 2. replace provider .so -------------------------------------------------
echo "Replacing CUDA EP with ROCm build ..."
cp "$ROCM_SO" "$TARGET"

# --- 2b. replace bundled libstdc++ with system version -----------------------
# The AppImage bundles an old libstdc++.so.6 (CXXABI up to 1.3.13) but the
# ROCm provider compiled on Gentoo requires CXXABI_1.3.15 (GCC 13+).
# libstdc++ is backward compatible, so upgrading is safe.
BUNDLED_STDCXX="$APPDIR/vv-engine/engine_internal/libstdc++.so.6"
if [[ -f "$BUNDLED_STDCXX" ]]; then
  SYS_STDCXX=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6.*x86-64/{print $NF}' | head -1)
  [[ -z "$SYS_STDCXX" ]] && SYS_STDCXX=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 /{print $NF}' | head -1)
  if [[ -n "$SYS_STDCXX" ]]; then
    echo "Replacing bundled libstdc++.so.6 with system version: $SYS_STDCXX"
    cp "$SYS_STDCXX" "$BUNDLED_STDCXX"
  else
    echo "WARNING: system libstdc++.so.6 not found via ldconfig; bundled version kept"
  fi
fi

# --- 2c. patch AppRun to set MIOPEN_FIND_MODE=FAST ---------------------------
# Without this, MIOpen runs exhaustive kernel timing on every new convolution
# input size (= new text length), causing multi-second first-call latency.
# FAST mode uses heuristic solver selection: no timing, no latency spike.
APPRUN="$APPDIR/AppRun"
if [[ -f "$APPRUN" ]]; then
  echo "Patching AppRun: adding MIOPEN_FIND_MODE=FAST ..."
  sed -i '/^apprun=/a export MIOPEN_FIND_MODE=FAST' "$APPRUN"
fi

# --- 3. repack squashfs ------------------------------------------------------
echo "Repacking squashfs (zstd, 128 KiB blocks) ..."
mksquashfs "$APPDIR" "$NEW_SFS" \
  -comp zstd \
  -b 131072 \
  -nopad \
  -no-progress

# --- 4. combine runtime + new squashfs ---------------------------------------
echo "Combining runtime + new squashfs ..."
dd if="$INPUT_APPIMAGE" of="$RUNTIME_BIN" bs="$RUNTIME_SIZE" count=1 status=none
cat "$RUNTIME_BIN" "$NEW_SFS" > "$OUTPUT_APPIMAGE"
chmod +x "$OUTPUT_APPIMAGE"

echo ""
echo "Done: $OUTPUT_APPIMAGE"
echo "Size: $(du -h "$OUTPUT_APPIMAGE" | cut -f1)"
