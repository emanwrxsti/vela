$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$stage = Join-Path $env:RUNNER_TEMP 'supr-meow-vela-0.1.6-windows-x86_64'
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $stage | Out-Null

$exe = Get-ChildItem build/windows -Filter supr-meow-vela.exe -Recurse | Select-Object -First 1
if (!$exe) { throw 'Native miner executable missing' }
Copy-Item $exe.FullName "$stage/supr-meow-vela.exe"

# CUDA runtime DLLs the miner links against (cudart + cuBLAS/cuBLASLt).
Get-ChildItem "$env:CUDA_PATH/bin" -Filter '*.dll' -Recurse |
    Where-Object { $_.Name -match '^(cudart|cublas|cublasLt)' } |
    Sort-Object Name -Unique | Copy-Item -Destination $stage

Copy-Item windows/* $stage
Copy-Item LICENSE,NOTICE $stage
Copy-Item RELEASE-v0.1.6.md $stage

$licenses = Join-Path $stage 'licenses'
New-Item -ItemType Directory -Force $licenses | Out-Null
Copy-Item licenses/nlohmann-json-LICENSE.txt $licenses
foreach ($component in @('cuda_cudart','libcublas','cccl')) {
    $notice = Join-Path $env:CUDA_PATH "licenses/nvidia/$component.txt"
    if (Test-Path $notice) { Copy-Item $notice "$licenses/NVIDIA-$component.txt" }
}
if (Test-Path "$env:CUDA_PATH/installed-components.json") {
    Copy-Item "$env:CUDA_PATH/installed-components.json" "$licenses/NVIDIA-components.json"
}

$info = @{
    version = '0.1.6'; source_commit = (& git rev-parse HEAD).Trim()
    ci_run = "https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID"
    platform = 'native-windows-x86_64'; cuda = '13.3'; architectures = @(80,86,89,90,100,120)
    cpu_selftest = 'passed'; windows_gpu_hardware_test = $false
}
$info | ConvertTo-Json | Set-Content "$stage/BUILD-INFO.json"

# cudart/cublasLt are load-time linked, so a clean --help proves every runtime DLL resolves.
Push-Location $stage
try {
    & ./supr-meow-vela.exe --help
    if ($LASTEXITCODE) { throw 'Packaged native executable cannot start' }
} finally { Pop-Location }

Get-ChildItem $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\','/')
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relative
} | Set-Content "$stage/SHA256SUMS.txt"

New-Item -ItemType Directory -Force dist | Out-Null
$zip = Join-Path $root 'dist/supr-meow-vela-0.1.6-windows-x86_64.zip'
Compress-Archive -Path "$stage/*" -DestinationPath $zip -Force
"{0}  {1}" -f (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant(), (Split-Path $zip -Leaf) | Set-Content "$zip.sha256"
Write-Host "Packaged $zip"
