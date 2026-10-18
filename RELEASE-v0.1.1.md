supr-meow-vela 0.1.1 fixes multi-GPU mining and includes performance improvements across supported NVIDIA cards.

## Changes

- Fixed CUDA `invalid argument` failures when one miner process uses multiple GPUs.
- Improved mining throughput and model loading.
- `--check` now verifies the consensus test vector on every selected GPU.
- Mining and benchmark failures now return a nonzero exit status.
- Docker includes the same qualified miner and CUDA runtime as the Linux package, with TLS support.

## Measured performance

Controlled pre-release comparisons of the included optimizations, using the same settings for each baseline and optimized run. Rates are inferences per second; results vary with clocks, cooling, power limits and workload.

| GPU | Batch × lanes | Power cap | Baseline inf/s | Optimized inf/s | Gain |
|---|---:|---:|---:|---:|---:|
| RTX 3070 | 4 × 2 | 240 W | 24.22 | 25.18 | +3.98% |
| RTX 4070 Ti SUPER | 4 × 2 | 285 W | 50.05 | 52.17 | +4.24% |
| RTX 5080 | 4 × 2 | 360 W | 73.61 | 75.89 | +3.10% |
| RTX 5090 | 4 × 2 | 600 W | 132.59 | 136.89 | +3.24% |
| CMP 170HX | 8 × 2 | 250 W | 37.53 | 40.02 | +6.62% |
| RTX PRO 6000 | 8 × 2 | 600 W | 136.54 | 141.33 | +3.50% |

These are measurements of the included optimization changes, not a fresh benchmark of every final release package.

## Downloads

| Platform | Asset |
|---|---|
| Linux / MMPOS / SMOS | `supr-meow-vela-0.1.1.tar.gz` |
| HiveOS custom miner | `supr-meow-vela-hiveos.tar.gz` |
| Native Windows x64 | `supr-meow-vela-0.1.1-windows-x86_64.zip` |

The Linux tarball includes `h-run.sh`, `mmpos-run.sh` and `smos-run.sh`. The separate HiveOS archive uses the directory name required by its custom-miner installer. Every archive includes a file checksum manifest; external `.sha256` files and `SHA256SUMS` verify the downloads.

Docker: `ocminersupr/supr-meow-vela:0.1.1` and `:latest`.

The model is unchanged and downloads automatically when needed. Existing `sd-turbo-int.poisd` files can be reused.

## Requirements and validation

- Six CUDA targets: SM 80, 86, 89, 90, 100 and 120, covering Ampere, Ada, Hopper and Blackwell.
- CUDA 13-capable NVIDIA driver (R580 or newer). Linux packages require glibc 2.38+ and OpenSSL 3; Docker provides the userspace dependencies.
- Exact consensus-output checks and live multi-GPU pool mining are validated on Linux.
- Native Windows builds pass CPU SHA-256/network tests and packaged executable loading in CI. Windows GPU mining is not hardware-tested on the hosted CI runner. Windows uses plain TCP; TLS is available on Linux and Docker.
