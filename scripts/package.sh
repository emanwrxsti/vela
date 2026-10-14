#!/usr/bin/env bash
# Build a release tarball: binary + bundled CUDA runtime libs + hiveos scripts. Run on the build host after `cmake --build build`.
set -euo pipefail
cd "$(dirname "$0")/.."
VER=${1:-0.1.0}; CUDA=${CUDA_HOME:-/usr/local/cuda-13.3}
D=dist/supr-meow-vela-$VER; rm -rf "$D"; mkdir -p "$D/lib"
cp build/supr-meow-vela "$D/"; strip "$D/supr-meow-vela"
cp "$CUDA/targets/x86_64-linux/lib/libcudart.so.13" "$CUDA/targets/x86_64-linux/lib/libcublasLt.so.13" "$D/lib/"
cp README.md "$D/"; cp hiveos/h-*.sh hiveos/h-manifest.conf "$D/"
( cd dist && tar czf supr-meow-vela-$VER.tar.gz supr-meow-vela-$VER && sha256sum supr-meow-vela-$VER.tar.gz > supr-meow-vela-$VER.tar.gz.sha256 )
# HiveOS wants the package dir named after the miner
( cd dist && rm -rf supr-meow-vela && cp -r supr-meow-vela-$VER supr-meow-vela && tar czf supr-meow-vela-hiveos.tar.gz supr-meow-vela && rm -rf supr-meow-vela )
ls -la dist/
