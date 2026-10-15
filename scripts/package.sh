#!/usr/bin/env bash
# Build the Linux release tarball: binary + bundled CUDA runtime libs + HiveOS/MMPOS/SMOS
# launchers. One tarball serves all three mining OSes (pick the matching *-run.sh).
# Run on the build host after `cmake --build <build-dir>`.
set -euo pipefail
cd "$(dirname "$0")/.."
VER=${1:-0.1.0}
BUILD=${BUILD_DIR:-build-release}
CUDA=${CUDA_HOME:-/usr/local/cuda-13.3}
D=dist/supr-meow-vela-$VER
rm -rf "$D"; mkdir -p "$D/lib"

cp "$BUILD/supr-meow-vela" "$D/"; strip "$D/supr-meow-vela"
cp "$CUDA/targets/x86_64-linux/lib/libcudart.so.13" "$CUDA/targets/x86_64-linux/lib/libcublasLt.so.13" "$D/lib/"
cp README.md LICENSE NOTICE "$D/"
# HiveOS
cp hiveos/h-config.sh hiveos/h-run.sh hiveos/h-stats.sh hiveos/h-manifest.conf "$D/"
# MMPOS / SMOS
cp mmpos/mmpos-run.sh smos/smos-run.sh "$D/"
chmod +x "$D"/*.sh "$D/supr-meow-vela"

# Inner manifest: sha256 of every shipped file (package self-check)
( cd "$D" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt )

# One tarball for all Linux mining OSes
( cd dist && tar czf "supr-meow-vela-$VER.tar.gz" "supr-meow-vela-$VER" \
    && sha256sum "supr-meow-vela-$VER.tar.gz" > "supr-meow-vela-$VER.tar.gz.sha256" )

# HiveOS also accepts a package dir named after the miner
( cd dist && rm -rf supr-meow-vela && cp -r "supr-meow-vela-$VER" supr-meow-vela \
    && tar czf supr-meow-vela-hiveos.tar.gz supr-meow-vela && rm -rf supr-meow-vela )

echo "== dist =="; ls -la dist/ | grep -E "$VER|hiveos"
echo "== package contents =="; tar tzf "dist/supr-meow-vela-$VER.tar.gz"
