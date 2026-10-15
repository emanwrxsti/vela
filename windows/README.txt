supr-meow-vela 0.1.0 - native Windows x64 / NVIDIA CUDA

Vela GPU miner (Proof-of-Inference v2, sd-turbo-int). Runs natively on Windows;
WSL and Docker are not required. Use Windows 10/11 x64 and an NVIDIA Ampere or
newer GPU (sm_80/86/89/120) with a driver compatible with CUDA 13.3. Install the
Microsoft Visual C++ 2015-2022 x64 runtime if Windows reports a missing
VCRUNTIME/MSVCP DLL.

1. Extract the complete ZIP to a writable directory.
2. Edit start.cmd and set WALLET to your own Vela wallet.
3. Double-click start.cmd. The sd-turbo-int model (~1.2 GB) downloads on first
   start and is cached in the models directory; keep it for later starts.

PowerShell examples:
  .\start.ps1 -Wallet YOUR_VELA_WALLET -Worker rig1 -Devices 0
  .\start.ps1 -Wallet YOUR_VELA_WALLET -Devices '0,1'          # two GPUs
  .\start.ps1 -Wallet YOUR_VELA_WALLET -Batch 8 -Lanes 2       # larger GPUs

It identifies itself to the pool as supr-meow-vela/0.1.0 and connects to
vela.suprnova.cc:7601 unless you pass -Pool. This Windows build has no TLS
(OpenSSL) support; use the plain-TCP port.

The CUDA share-target math and the native sockets are validated in GitHub CI.
The hosted Windows runner has no NVIDIA GPU, so Windows GPU mining is not
hardware-tested there; Linux release packages are tested mining live.

BUILD-INFO.json records the source commit and CI run. SHA256SUMS.txt covers the
files in this package; the external .sha256 file covers the complete ZIP.
Source and build instructions: https://github.com/ocminer/supr-meow-vela
