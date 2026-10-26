supr-meow-vela 0.1.6 increases mining throughput while preserving exact consensus outputs.

## Measured performance

Repeated, matched Linux benchmarks against v0.1.5, using the same GPU, model, batch size, lane count and power cap. Rates are inferences per second (inf/s).

| GPU | Batch × lanes | Power cap | v0.1.5 inf/s | v0.1.6 inf/s | Increase |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 4 × 2 | 600 W | 138.78 | 152.04 | **+9.6%** |

Results depend on clocks, power limits, cooling and workload; the table is a controlled benchmark, not a guarantee of pool-reported speed. Other supported GPUs were not re-benchmarked for this release.

## What changed

Faster attention, normalization and GEMM kernels. All arithmetic is bit-identical to the reference; every change is validated against exact model outputs. Default batch and lane settings are unchanged.

## Validation

- Exact model-output checks (12 seeds, batch 1 and 4), independent integer-operation tests (LayerNorm/GroupNorm oracles, attention score tests), attention edge cases (17 cases incl. causal and padded keys) and the spec test vector passed on RTX 5090 (SM 120); the test vector and attention edge cases also passed on CMP 170HX (SM 80) and RTX 3070 (SM 86).
- Linux CPU, TCP reconnect and TLS reconnect regression tests passed.
- Live mining accepted shares without rejects on the tested RTX 5090.
- Native Windows CI builds all six CUDA targets, runs CPU/network/reconnect tests and checks that the packaged executable starts. Windows GPU mining is not hardware-tested on the hosted runner.

## Downloads

| Platform | Asset |
|---|---|
| Linux / MMPOS / SMOS | `supr-meow-vela-0.1.6.tar.gz` |
| HiveOS custom miner | `supr-meow-vela-hiveos.tar.gz` |
| Native Windows x64 | `supr-meow-vela-0.1.6-windows-x86_64.zip` |

The Linux tarball includes HiveOS, MMPOS and SMOS launchers. The separate HiveOS archive has the directory name required by its installer. Packages include file checksums; `.sha256` sidecars and `SHA256SUMS` verify downloads.

Docker: `ocminersupr/supr-meow-vela:0.1.6` and `:latest`.

All builds target SM 80, 86, 89, 90, 100 and 120 (Ampere, Ada, Hopper and Blackwell) with CUDA 13.3. A CUDA 13-capable NVIDIA driver (R580+) is required. Linux archives require glibc 2.38+ and OpenSSL 3; Docker provides the userspace dependencies. Windows uses plain TCP; TLS is available on Linux and Docker.

Reuse the existing `sd-turbo-int.poisd` model. HiveOS downloads from `https://vela.suprnova.cc/dl/sd-turbo-int.poisd` first and uses `https://www.suprnova.cc/dl/sd-turbo-int.poisd` as its verified fallback.

## License

This release uses the **Suprnova Vela Source-Available License 1.0**. Running the unmodified miner or verifier, including for profit, is permitted. **Incorporation into a commercial or closed-source miner requires prior written permission from ocminer / suprnova.** Permitted derivative miners must be non-commercial and publish complete corresponding source under the same license. Read the included `LICENSE` and `NOTICE` for the full terms.

Previously released versions through v0.1.4 retain their original MIT terms. Third-party components and models keep their own licenses.
