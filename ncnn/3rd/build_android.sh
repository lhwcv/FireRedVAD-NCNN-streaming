#!/usr/bin/env sh

# Minimal Android NCNN build (32-bit only: armeabi-v7a)
# Usage: ./3rd/build_android.sh [version] [ANDROID_NDK]
# - version default: 20260113
# - ANDROID_NDK: optional if $ANDROID_NDK env is set

VER="${1:-20260113}"
NDK="${2:-${ANDROID_NDK:-}}"
if [ -z "${NDK}" ]; then
  echo "Set ANDROID_NDK or pass it as the 2nd argument." >&2
  exit 1
fi

ABI="armeabi-v7a"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$BASE_DIR/source/ncnn-$VER"
ZIP="$BASE_DIR/source/ncnn-$VER.zip"
URL="https://github.com/Tencent/ncnn/archive/refs/tags/$VER.zip"
BUILD_DIR="$BASE_DIR/ncnn_build_android-$ABI"
PREFIX="$BASE_DIR/prebuild/Android/$ABI/ncnn"
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"

mkdir -p "$BASE_DIR/source"

if [ ! -d "$SRC_DIR" ]; then
  echo "[ncnn-android] Source not found: $SRC_DIR"
  if [ ! -f "$ZIP" ]; then
    echo "[ncnn-android] Downloading $URL -> $ZIP (curl/wget if available)"
    if command -v curl >/dev/null 2>&1; then
      curl -fL "$URL" -o "$ZIP" || { echo "Download failed; place zip at $ZIP"; exit 1; }
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$ZIP" "$URL" || { echo "Download failed; place zip at $ZIP"; exit 1; }
    else
      echo "No curl/wget found. Manually download to: $ZIP"; exit 1
    fi
  fi
  echo "[ncnn-android] Unzipping $ZIP"
  unzip -q -o "$ZIP" -d "$BASE_DIR/source"
fi

rm -rf "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM=android-21 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DNCNN_PLATFORM_API=OFF \
  -DNCNN_OPENMP=OFF \
  -DNCNN_VULKAN=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_BENCHMARK=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"

cmake --build "$BUILD_DIR" -- -j"${JOBS:-$(getconf _NPROCESSORS_ONLN || echo 4)}"
cmake --install "$BUILD_DIR"

echo "[ncnn-android] Installed to: $PREFIX"
echo "[ncnn-android] Library:      $PREFIX/lib/libncnn.a"
echo "[ncnn-android] Headers:      $PREFIX/include/"
echo "[ncnn-android] Set NCNN_ROOT to this prefix when cross-compiling:"
echo "                 export NCNN_ROOT=\"$PREFIX\""
