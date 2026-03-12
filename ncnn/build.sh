#!/usr/bin/env sh

# Build FireRedVAD (NCNN backend) without `cmake --install`.
# Copies the built library and executable into ncnn/out.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
OUT_DIR="$SCRIPT_DIR/out"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [ "${1:-}" = "clean" ]; then
  rm -rf "$BUILD_DIR" "$OUT_DIR"
  echo "[clean] Removed $BUILD_DIR and $OUT_DIR"
  exit 0
fi

# If NCNN_ROOT is not set, try to infer from 3rd/prebuild
if [ -z "${NCNN_ROOT:-}" ]; then
  UNAME_S="$(uname -s)"
  case "$UNAME_S" in
    Darwin) PRE_OS="MacOS";;
    Linux)  PRE_OS="Linux";;
    *)      PRE_OS="Linux";;
  esac
  CANDIDATE="$SCRIPT_DIR/3rd/prebuild/$PRE_OS/ncnn"
  if [ -d "$CANDIDATE" ]; then
    export NCNN_ROOT="$CANDIDATE"
  else
    echo "[warn] NCNN_ROOT not set and $CANDIDATE not found."
    echo "build ncnn from source.."
    ( cd "$SCRIPT_DIR/3rd/" && sh ./build.sh )
    # after build, set NCNN_ROOT if expected path exists
    if [ -d "$CANDIDATE" ]; then
      export NCNN_ROOT="$CANDIDATE"
    else
      echo "[error] NCNN_ROOT not set and built prebuild not found at $CANDIDATE" >&2
      echo "        Please set NCNN_ROOT manually or re-run ./3rd/build.sh" >&2
      exit 1
    fi
  fi
fi

echo "[info] NCNN_ROOT=${NCNN_ROOT:-unset}"
echo "[info] Configuring ($BUILD_TYPE) ..."
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "[info] Building ..."
cmake --build "$BUILD_DIR" -- -j"${JOBS:-$(getconf _NPROCESSORS_ONLN || echo 4)}"

echo "[info] Collecting artifacts to $OUT_DIR ..."
mkdir -p "$OUT_DIR"

# Copy libs
if ls "$BUILD_DIR"/libfirered_vad_stream.* >/dev/null 2>&1; then
  cp "$BUILD_DIR"/libfirered_vad_stream.* "$OUT_DIR"/
fi

# Copy executables
for name in test_vad_stream test_vad_non_stream test_aed_non_stream; do
  if [ -x "$BUILD_DIR/$name" ]; then
    cp "$BUILD_DIR/$name" "$OUT_DIR"/
  fi
done
