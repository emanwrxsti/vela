@echo off
setlocal
rem Enter your Vela wallet below, then double-click this file.
set "WALLET=YOUR_VELA_WALLET"
set "WORKER=windows"
set "DEVICES=0"
if "%WALLET%"=="YOUR_VELA_WALLET" (
  echo Edit start.cmd and set WALLET to your own Vela address first.
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Wallet "%WALLET%" -Worker "%WORKER%" -Devices "%DEVICES%"
pause
