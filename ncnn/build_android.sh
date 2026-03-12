#!/usr/bin/env sh

# Cross-build runtime/ncnn for Android and place artifacts into out_android.
# Defaults:
#   - ABI: armeabi-v7a (32-bit)
#   - ANDROID_PLATFORM: 21
#   - NCNN_ROOT: inferred from ./3rd/prebuild/Android/<abi>/ncnn if not set

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ABI="${ABI:-armeabi-v7a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-21}"
BUILD_DIR="$SCRIPT_DIR/build-android-$ABI"
OUT_DIR="$SCRIPT_DIR/out_android/$ABI"

NDK="${1:-${ANDROID_NDK:-}}"
if [ -z "${NDK}" ]; then
  echo "Set ANDROID_NDK env or pass as first arg." >&2
  exit 1
fi

# Infer NCNN_ROOT if not provided
if [ -z "${NCNN_ROOT:-}" ]; then
  CANDIDATE="$SCRIPT_DIR/3rd/prebuild/Android/$ABI/ncnn"
  if [ -d "$CANDIDATE" ]; then
    export NCNN_ROOT="$CANDIDATE"
  else
    echo "[info] NCNN_ROOT not set; building NCNN into $CANDIDATE ..."
    (
      cd "$SCRIPT_DIR/3rd" && sh ./build_android.sh 20260113 "$NDK"
    )
    if [ -d "$CANDIDATE" ]; then
      export NCNN_ROOT="$CANDIDATE"
    else
      echo "[error] Failed to prepare NCNN at $CANDIDATE" >&2
      exit 1
    fi
  fi
fi

echo "[info] ANDROID_NDK=$NDK"
echo "[info] ABI=$ABI, ANDROID_PLATFORM=android-$ANDROID_PLATFORM"
echo "[info] NCNN_ROOT=${NCNN_ROOT}"

rm -rf "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM=android-$ANDROID_PLATFORM \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -- -j"${JOBS:-$(getconf _NPROCESSORS_ONLN || echo 4)}"

mkdir -p "$OUT_DIR"
echo "[info] Copying artifacts to $OUT_DIR"

# Copy shared lib
if ls "$BUILD_DIR"/libfirered_vad_stream.* >/dev/null 2>&1; then
  cp "$BUILD_DIR"/libfirered_vad_stream.* "$OUT_DIR"/
fi

# Copy executables
for name in test_vad_stream test_vad_non_stream test_aed_non_stream; do
  if [ -f "$BUILD_DIR/$name" ]; then
    cp "$BUILD_DIR/$name" "$OUT_DIR"/
  elif [ -f "$BUILD_DIR/$name$EXEEXT" ]; then
    cp "$BUILD_DIR/$name$EXEEXT" "$OUT_DIR"/
  fi
done

echo "[done] Android artifacts are in: $OUT_DIR"
