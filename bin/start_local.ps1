param(
    [switch]$SkipInstall,
    [int]$StartupTimeoutSec = 60,
    [int]$PostStartMonitorSec = 15,
    [string]$ResultsFile = 'STARTUP_RESULTS.md'
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$requirementsFile = Join-Path $repoRoot 'requirements.txt'
$entryScript = Join-Path $repoRoot 'sec_cam.py'
$resultsPath = Join-Path $repoRoot $ResultsFile
$runtimeLogDir = Join-Path $repoRoot 'logs'
$stdoutLog = Join-Path $runtimeLogDir 'bunnycam-start.stdout.log'
$stderrLog = Join-Path $runtimeLogDir 'bunnycam-start.stderr.log'

function Resolve-PythonExe {
    param([string]$VirtualEnvPython)

    if (Test-Path $VirtualEnvPython) {
        return $VirtualEnvPython
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Python was not found. Create .venv or add python to PATH.'
    }

    return $pythonCommand.Source
}

function Test-BunnyCamStatus {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 3
    )

    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSeconds
        return @{ Ok = $true; Payload = $response; Error = $null }
    } catch {
        return @{ Ok = $false; Payload = $null; Error = $_.Exception.Message }
    }
}

function Get-LogTail {
    param(
        [string]$Path,
        [int]$Lines = 20
    )

    if (-not (Test-Path $Path)) {
        return ''
    }

    $content = Get-Content -Path $Path -Tail $Lines -ErrorAction SilentlyContinue
    if (-not $content) {
        return ''
    }

    return ($content -join "`n")
}

function Ensure-ResultsFile {
    param([string]$Path)

    if (Test-Path $Path) {
        return
    }

    $initialContent = @(
        '# BunnyCam Startup Results',
        '',
        'This file is updated by `start_local.ps1` during each startup attempt.',
        '',
        'Each entry records:',
        '- local timestamp',
        '- hostname',
        '- backend, host, and port',
        '- whether startup succeeded or failed',
        '- PID or error details for follow-up hardening',
        ''
    )
    Set-Content -Path $Path -Value $initialContent -Encoding utf8
}

function Add-ResultsEntry {
    param(
        [string]$Path,
        [string]$Outcome,
        [string]$Details,
        [string]$Url,
        [string]$Backend,
        [string]$BindHost,
        [int]$Port,
        [Nullable[int]]$ProcessId,
        [string]$StatusSummary
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $hostname = [System.Net.Dns]::GetHostName()
    $pidText = if ($null -ne $ProcessId) { $ProcessId.ToString() } else { 'n/a' }

    $entry = @(
        '',
        "## $timestamp | $hostname | $Outcome",
        '',
        "- Timestamp: $timestamp",
        "- Hostname: $hostname",
        "- Backend: $Backend",
        "- Bind Host: $BindHost",
        "- Port: $Port",
        "- URL: $Url",
        "- PID: $pidText",
        "- Summary: $StatusSummary",
        "- Details: $Details",
        ''
    )

    Add-Content -Path $Path -Value $entry -Encoding utf8
}

$pythonExe = Resolve-PythonExe -VirtualEnvPython $venvPython

if (-not $env:CAMERA_BACKEND) {
    $env:CAMERA_BACKEND = 'laptop'
}

if (-not $env:BUNNYCAM_PORT) {
    $env:BUNNYCAM_PORT = '8001'
}

if (-not $env:BUNNYCAM_HOST) {
    $env:BUNNYCAM_HOST = '127.0.0.1'
}

$backend = $env:CAMERA_BACKEND
$bindHost = $env:BUNNYCAM_HOST
$port = [int]$env:BUNNYCAM_PORT
$statusUrl = "http://$bindHost`:$port/status"
$rootUrl = "http://$bindHost`:$port/"

New-Item -ItemType Directory -Path $runtimeLogDir -Force | Out-Null
Ensure-ResultsFile -Path $resultsPath

$existingStatus = Test-BunnyCamStatus -Url $statusUrl
if ($existingStatus.Ok) {
    $summary = 'Endpoint already healthy; no new process started.'
    Add-ResultsEntry -Path $resultsPath -Outcome 'success' -Details $summary -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $null -StatusSummary $summary
    Write-Host "BunnyCam is already responding at $rootUrl"
    exit 0
}

if (-not $SkipInstall) {
    Write-Host 'Installing or verifying Python requirements...'
    & $pythonExe -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        $summary = 'Requirement installation failed.'
        $details = Get-LogTail -Path $stderrLog
        if (-not $details) {
            $details = 'pip install -r requirements.txt returned a non-zero exit code.'
        }
        Add-ResultsEntry -Path $resultsPath -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $null -StatusSummary $summary
        throw 'Unable to install requirements.'
    }
}

if (Test-Path $stdoutLog) {
    Remove-Item $stdoutLog -Force
}

if (Test-Path $stderrLog) {
    Remove-Item $stderrLog -Force
}

Write-Host "Starting BunnyCam on $rootUrl using backend '$backend'"
$process = Start-Process -FilePath $pythonExe -ArgumentList $entryScript -WorkingDirectory $repoRoot -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

$deadline = (Get-Date).AddSeconds($StartupTimeoutSec)
$healthyPayload = $null

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2

    if ($process.HasExited) {
        $summary = "BunnyCam exited before startup completed (exit code $($process.ExitCode))."
        $stderrTail = Get-LogTail -Path $stderrLog
        $stdoutTail = Get-LogTail -Path $stdoutLog
        $details = if ($stderrTail) { $stderrTail } elseif ($stdoutTail) { $stdoutTail } else { 'No startup output captured.' }
        Add-ResultsEntry -Path $resultsPath -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary
        throw $summary
    }

    $statusResult = Test-BunnyCamStatus -Url $statusUrl
    if ($statusResult.Ok) {
        $healthyPayload = $statusResult.Payload
        break
    }
}

if (-not $healthyPayload) {
    try {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }

    $summary = "BunnyCam did not become healthy within $StartupTimeoutSec seconds."
    $stderrTail = Get-LogTail -Path $stderrLog
    $stdoutTail = Get-LogTail -Path $stdoutLog
    $details = if ($stderrTail) { $stderrTail } elseif ($stdoutTail) { $stdoutTail } else { 'No startup output captured before timeout.' }
    Add-ResultsEntry -Path $resultsPath -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary
    throw $summary
}

$monitorDeadline = (Get-Date).AddSeconds($PostStartMonitorSec)
while ((Get-Date) -lt $monitorDeadline) {
    Start-Sleep -Seconds 2

    if ($process.HasExited) {
        $summary = "BunnyCam became healthy, then exited during the monitor window (exit code $($process.ExitCode))."
        $stderrTail = Get-LogTail -Path $stderrLog
        $stdoutTail = Get-LogTail -Path $stdoutLog
        $details = if ($stderrTail) { $stderrTail } elseif ($stdoutTail) { $stdoutTail } else { 'No output captured during post-start monitoring.' }
        Add-ResultsEntry -Path $resultsPath -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary
        throw $summary
    }

    $statusResult = Test-BunnyCamStatus -Url $statusUrl
    if (-not $statusResult.Ok) {
        $summary = 'BunnyCam process stayed alive but /status stopped responding during the monitor window.'
        $details = $statusResult.Error
        Add-ResultsEntry -Path $resultsPath -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary
        throw $summary
    }
}

$runtimeSummary = "Healthy on /status with runtime_initialized=$($healthyPayload.runtime_initialized) and backend=$($healthyPayload.backend)."
Add-ResultsEntry -Path $resultsPath -Outcome 'success' -Details $runtimeSummary -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary 'BunnyCam started successfully and passed the monitor window.'

Write-Host "BunnyCam is healthy at $rootUrl"
Write-Host "PID: $($process.Id)"
Write-Host "Results log: $resultsPath"
Write-Host "Stdout log: $stdoutLog"
Write-Host "Stderr log: $stderrLog"