# supr-meow-vela

GPU miner for **Vela** (Proof-of-Inference v2, `sd-turbo-int` W8A16) and the pool-side share verifier for vela.suprnova.cc.
Pure C++/CUDA: no Python, no torch. Bit-exact against `contrib/poi-sd/sdint.py` (the consensus reference) — every share
is a real inference, verified by the pool with the same engine in `--verify-server` mode.

* int8 tensor-core GEMMs through cuBLASLt (16-bit activations as a hi/lo int8 split with exact int64 reconstruction)
* fused exact attention (`mma.sync` int8, 3-pass softmax, never materialises the score matrix)
* exact integer GroupNorm / LayerNorm / LUT non-linearities, ChaCha20 noise on the GPU, CUDA Graphs, 2 lanes per GPU
* one fatbin for Ampere (sm_80 CMP 170HX / A100 / 30xx, sm_86 A6000), Ada (sm_89), Hopper (sm_90 H100 / H200), Blackwell datacenter (sm_100 B200) and Blackwell consumer (sm_120 RTX 50xx / PRO 6000) — needs CUDA 13 to build the full set

## Download
[GitHub Releases](https://github.com/ocminer/supr-meow-vela/releases) (and mirrored at https://vela.suprnova.cc/dl/):

* `supr-meow-vela-0.1.3.tar.gz` — Linux x86-64 + bundled CUDA 13.3 runtime. One tarball serves **HiveOS** (`h-run.sh`), **MMPOS** (`mmpos-run.sh`) and **SMOS** (`smos-run.sh`) — pick the launcher for your OS; set `POOL_URL` and `WALLET` in the flight sheet. The model downloads automatically if it is not cached.
* `supr-meow-vela-0.1.3-windows-x86_64.zip` — native **Windows** x64 build (no WSL/Docker). Edit `start.cmd` with your wallet and run it; the model downloads on first start. No TLS on Windows — use the plain-TCP port.
* `sd-turbo-int.poisd` — the model (~1.2 GB), plus `SHA256SUMS`.

The model's primary URL is `https://vela.suprnova.cc/dl/sd-turbo-int.poisd`;
`https://www.suprnova.cc/dl/sd-turbo-int.poisd` is its mirror. SHA-256:
`ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36`.
For the published HiveOS 0.1.1 package, set `MODEL_URL` to either URL in the
flight sheet's extra config; prefer the Vela URL.

The HiveOS launcher (0.1.2 and newer) defaults to the Vela URL and tries the
Suprnova mirror only if downloading or verifying the primary fails. Downloads
are checksum-verified before caching; cached files are checked before mining.
An explicit `MODEL_PATH` reuses a local file. `MODEL_URL` overrides the primary;
for a custom URL, set `MODEL_FALLBACK_URL` explicitly if a mirror is wanted.
Set `MODEL_FALLBACK_URL=""` to disable fallback. `MODEL_SHA256` can override the
expected checksum for an explicitly selected model.

## Usage
```
./supr-meow-vela -o vela.suprnova.cc:7601 -u vela1…yourAddress.worker --model sd-turbo-int.poisd [--batch 8 --lanes 2] [--devices 0,1] [--tls]
./supr-meow-vela --check                  # spec test vector on every selected GPU (sha256(x0) = bef682cc…)
./supr-meow-vela --bench 20 --batch 8     # throughput
./supr-meow-vela --verify-server --port 9010 --mem 1024 --batch 1   # pool verifier (HTTP POST /verify, GET /health)
```
Measured (batch 8 × 2 lanes): RTX 5090 / RTX PRO 6000 ≈ 125–135 inf/s (575–600 W); RTX 6000 Ada ≈ 51; H200 ≈ 92; B200 ≈ 113; CMP 170HX ≈ 31.

## Build
CUDA 12.8+ (13.3 used for releases), CMake ≥ 3.24, gcc 13. `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`.
Release tarballs ship `lib/libcudart.so.13` and `lib/libcublasLt.so.13` next to the binary (RPATH `$ORIGIN/lib`); the NVIDIA driver must be ≥ R580 (CUDA 13).
The Linux packages require glibc 2.38 or newer and OpenSSL 3. The Docker image
includes a compatible userspace for hosts with an older distribution.

Build with `-DMEOW_BUILD_TESTS=ON` and run `ctest --test-dir build --output-on-failure` for CPU tests.
With two available GPUs and the model installed, run
`python3 tests/multi_gpu.py --miner ./build/supr-meow-vela --model sd-turbo-int.poisd --devices 0,1`.
This checks the consensus vector across devices in both orders, eager and graph execution, and concurrent mining lanes.

## Protocol
Named-params JSON-RPC stratum, see `VELA-STRATUM-PROTOCOL.md` in the pool repo (`mining.subscribe` / `mining.authorize` /
`mining.notify` with `prompt_tokens` / `mining.submit` with `x0_hex` + `hashpoi_hex`; difficulty = expected inferences per share).
