supr-meow-vela 0.1.3 fixes pool-connection handling. Miners on the TLS port could lose their connection repeatedly and log `stratum: submit failed (not connected)`. Mining kernels, model and HiveOS setup are unchanged from 0.1.2.

## Changes

- TLS connections are now stable. Earlier versions read from and wrote to the same TLS session from different threads at the same time, which OpenSSL does not allow; under load this corrupted the session and dropped the connection. All TLS I/O is now serialised and reads are non-blocking, so submitting a share never waits for the reader.
- Reconnecting can no longer crash the miner. The connection was torn down while a share could still be being written to it.
- A write to a connection the pool has already closed no longer terminates the miner on Linux (SIGPIPE); it reconnects instead.
- A failed write now triggers an immediate reconnect instead of waiting for the operating system to notice a dead connection, which could take hours.
- Liveness check: after 2 minutes without any message from the pool the miner sends a keepalive ping, and it reconnects after 5 minutes of silence.
- Shares found while reconnecting, or for a job from the previous connection, are discarded with a clear log line (`share discarded …`) instead of being sent. Such shares are invalid after a reconnect and would only have been rejected.

## Downloads

| Platform | Asset |
|---|---|
| Linux / MMPOS / SMOS | `supr-meow-vela-0.1.3.tar.gz` |
| HiveOS custom miner | `supr-meow-vela-hiveos.tar.gz` |
| Native Windows x64 | `supr-meow-vela-0.1.3-windows-x86_64.zip` |

The Linux tarball includes `h-run.sh`, `mmpos-run.sh` and `smos-run.sh`. The separate HiveOS archive uses the directory name required by its custom-miner installer. Every archive includes a file checksum manifest; external `.sha256` files and `SHA256SUMS` verify the downloads.

Docker: `ocminersupr/supr-meow-vela:0.1.3` and `:latest`.

The model is unchanged (`sd-turbo-int.poisd`, SHA-256 `ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36`) and downloads automatically when needed.

## Requirements and validation

- Six CUDA targets: SM 80, 86, 89, 90, 100 and 120, covering Ampere, Ada, Hopper and Blackwell.
- CUDA 13-capable NVIDIA driver (R580 or newer). Linux packages require glibc 2.38+ and OpenSSL 3; Docker provides the userspace dependencies.
- Validated on Linux with live pool mining over both plain TCP and TLS, including forced connection drops during mining.
- Native Windows builds pass CPU SHA-256/network tests and packaged executable loading in CI. Windows uses plain TCP; TLS is available on Linux and Docker.
