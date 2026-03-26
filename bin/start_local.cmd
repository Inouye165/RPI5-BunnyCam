@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if "%CAMERA_BACKEND%"=="" set "CAMERA_BACKEND=laptop"
if "%BUNNYCAM_PORT%"=="" set "BUNNYCAM_PORT=8001"
if "%BUNNYCAM_HOST%"=="" set "BUNNYCAM_HOST=127.0.0.1"

echo Starting BunnyCam on http://127.0.0.1:%BUNNYCAM_PORT% using backend "%CAMERA_BACKEND%"
"%PYTHON_EXE%" "%REPO_ROOT%\sec_cam.py"