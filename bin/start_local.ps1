$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Python was not found. Create .venv or add python to PATH.'
    }
    $pythonExe = $pythonCommand.Source
}

if (-not $env:CAMERA_BACKEND) {
    $env:CAMERA_BACKEND = 'laptop'
}

if (-not $env:BUNNYCAM_PORT) {
    $env:BUNNYCAM_PORT = '8001'
}

Write-Host "Starting BunnyCam on http://127.0.0.1:$($env:BUNNYCAM_PORT) using backend '$($env:CAMERA_BACKEND)'"
& $pythonExe (Join-Path $repoRoot 'sec_cam.py')