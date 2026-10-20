supr-meow-vela 0.1.4 completes the pool reconnection repair introduced in 0.1.3. It prevents in-flight work from the old connection being submitted on a new one, including when the pool reuses job IDs.

## Changes

- A share's connection is checked atomically with its submission, closing a remaining reconnect race in 0.1.3.
- Every job retains its original connection identity and extranonce, so reconnects cannot mix new connection state with old work.
- TLS session access remains serialized; failed writes wake the reader for reconnect, SIGPIPE cannot terminate the Linux miner, and idle connections retain the two-minute ping and five-minute reconnect limits from 0.1.3.
- Automated TCP and TLS regression tests use eight concurrent submitters, 24 forced disconnects per transport, and repeated job IDs to check that old-session work is discarded.
- Mining kernels, model bytes and HiveOS download fallback are unchanged. Existing models can be reused.

## Measured performance

Historical controlled measurements of the optimizations first released in 0.1.1, using the same settings for each baseline and optimized run. The mining kernels are unchanged in 0.1.4; this table does not represent a speed increase over 0.1.1. Rates are inferences per second; results vary with clocks, cooling, power limits and workload.

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
| Linux / MMPOS / SMOS | `supr-meow-vela-0.1.4.tar.gz` |
| HiveOS custom miner | `supr-meow-vela-hiveos.tar.gz` |
| Native Windows x64 | `supr-meow-vela-0.1.4-windows-x86_64.zip` |

The Linux tarball includes `h-run.sh`, `mmpos-run.sh` and `smos-run.sh`. The separate HiveOS archive uses the directory name required by its custom-miner installer. Every archive includes a file checksum manifest; external `.sha256` files and `SHA256SUMS` verify the downloads.

Docker: `ocminersupr/supr-meow-vela:0.1.4` and `:latest`.

The model is unchanged and downloads automatically when needed. Existing `sd-turbo-int.poisd` files can be reused.

## Requirements and validation

- Six CUDA targets: SM 80, 86, 89, 90, 100 and 120, covering Ampere, Ada, Hopper and Blackwell.
- CUDA 13-capable NVIDIA driver (R580 or newer). Linux packages require glibc 2.38+ and OpenSSL 3; Docker provides the userspace dependencies.
- Linux release validation includes consensus-output checks and live TLS mining through forced reconnects. The unchanged CUDA kernels retain the earlier multi-GPU qualification.
- Native Windows builds pass CPU SHA-256/network and reconnect tests and packaged executable loading in CI. Windows GPU mining is not hardware-tested on the hosted CI runner. Windows uses plain TCP; TLS is available on Linux and Docker.
