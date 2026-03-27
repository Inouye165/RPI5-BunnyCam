param(
    [switch]$SkipInstall,
    [int]$StartupTimeoutSec = 60,
    [int]$PostStartMonitorSec = 15,
    [string]$ResultsFile = 'STARTUP_RESULTS.md',
    [string]$Actor = 'manual',
    [string]$Issue,
    [string]$Fix,
    [string]$Note
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$commonScript = Join-Path $PSScriptRoot 'startup_common.ps1'
. $commonScript
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$requirementsFile = Join-Path $repoRoot 'requirements.txt'
$entryScript = Join-Path $repoRoot 'sec_cam.py'
$resultsPath = Join-Path $repoRoot $ResultsFile
$runtimeLogDir = Join-Path $repoRoot 'logs'
$stdoutLog = Join-Path $runtimeLogDir 'bunnycam-start.stdout.log'
$stderrLog = Join-Path $runtimeLogDir 'bunnycam-start.stderr.log'
$runtimeStatePath = Join-Path $runtimeLogDir 'bunnycam-runtime.json'
$managedComponents = @('BunnyCam Python web process')


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
$versions = Get-DependencyVersions -PythonExe $pythonExe
$lifecycleLines = Get-LifecycleSectionLines -BindHost $bindHost -Port $port -RuntimeStatePath $runtimeStatePath -StdoutLog $stdoutLog -StderrLog $stderrLog
$versionLines = Get-VersionSectionLines -Versions $versions
Set-ResultsFileSections -Path $resultsPath -LifecycleLines $lifecycleLines -VersionLines $versionLines

$existingStatus = Test-BunnyCamStatus -Url $statusUrl
if ($existingStatus.Ok) {
    $summary = 'Endpoint already healthy; no new process started.'
    $existingProcess = Get-BunnyCamProcessFromPort -Port $port
    $existingPid = if ($existingProcess) { [int]$existingProcess.ProcessId } else { $null }
    if ($null -ne $existingPid) {
        Write-RuntimeState -Path $runtimeStatePath -ProcessId $existingPid -Backend $backend -BindHost $bindHost -Port $port -Url $rootUrl -Actor $Actor -ManagedComponents $managedComponents
    }
    Add-ResultsEntry -Path $resultsPath -Action 'start' -Outcome 'success' -Details $summary -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $existingPid -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
    Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
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
        Add-ResultsEntry -Path $resultsPath -Action 'start' -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $null -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
        Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
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
        Add-ResultsEntry -Path $resultsPath -Action 'start' -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
        Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
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
    Add-ResultsEntry -Path $resultsPath -Action 'start' -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
    Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
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
        Add-ResultsEntry -Path $resultsPath -Action 'start' -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
        Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
        throw $summary
    }

    $statusResult = Test-BunnyCamStatus -Url $statusUrl
    if (-not $statusResult.Ok) {
        $summary = 'BunnyCam process stayed alive but /status stopped responding during the monitor window.'
        $details = $statusResult.Error
        Add-ResultsEntry -Path $resultsPath -Action 'start' -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
        Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
        throw $summary
    }
}

$runtimeSummary = "Healthy on /status with runtime_initialized=$($healthyPayload.runtime_initialized) and backend=$($healthyPayload.backend)."
Write-RuntimeState -Path $runtimeStatePath -ProcessId $process.Id -Backend $backend -BindHost $bindHost -Port $port -Url $rootUrl -Actor $Actor -ManagedComponents $managedComponents
Add-ResultsEntry -Path $resultsPath -Action 'start' -Outcome 'success' -Details $runtimeSummary -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $process.Id -StatusSummary 'BunnyCam started successfully and passed the monitor window.' -Actor $Actor -ManagedComponents $managedComponents
Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note

Write-Host "BunnyCam is healthy at $rootUrl"
Write-Host "PID: $($process.Id)"
Write-Host "Results log: $resultsPath"
Write-Host "Stdout log: $stdoutLog"
Write-Host "Stderr log: $stderrLog"