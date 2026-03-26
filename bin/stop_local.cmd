@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_local.ps1" %*
exit /b %ERRORLEVEL%
