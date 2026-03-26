param(
    [string]$ResultsFile = 'STARTUP_RESULTS.md',
    [string]$Actor = 'manual',
    [string]$Issue,
    [string]$Fix,
    [string]$Note
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'startup_common.ps1')

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runtimeLogDir = Join-Path $repoRoot 'logs'
$resultsPath = Join-Path $repoRoot $ResultsFile
$runtimeStatePath = Join-Path $runtimeLogDir 'bunnycam-runtime.json'
$stdoutLog = Join-Path $runtimeLogDir 'bunnycam-start.stdout.log'
$stderrLog = Join-Path $runtimeLogDir 'bunnycam-start.stderr.log'
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
$rootUrl = "http://$bindHost`:$port/"

New-Item -ItemType Directory -Path $runtimeLogDir -Force | Out-Null
$versions = Get-DependencyVersions -PythonExe $pythonExe
$lifecycleLines = Get-LifecycleSectionLines -BindHost $bindHost -Port $port -RuntimeStatePath $runtimeStatePath -StdoutLog $stdoutLog -StderrLog $stderrLog
$versionLines = Get-VersionSectionLines -Versions $versions
Set-ResultsFileSections -Path $resultsPath -LifecycleLines $lifecycleLines -VersionLines $versionLines

$runtimeState = Read-RuntimeState -Path $runtimeStatePath
$processId = $null

if ($runtimeState -and $runtimeState.process_id) {
    $processId = [int]$runtimeState.process_id
}

if ($null -eq $processId) {
    $fallbackProcess = Get-BunnyCamProcessFromPort -Port $port
    if ($fallbackProcess) {
        $processId = [int]$fallbackProcess.ProcessId
    }
}

if ($null -eq $processId) {
    $summary = 'No BunnyCam process was found to stop.'
    Add-ResultsEntry -Path $resultsPath -Action 'stop' -Outcome 'failure' -Details $summary -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $null -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
    Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
    throw $summary
}

try {
    $process = Get-Process -Id $processId -ErrorAction Stop
    Stop-Process -Id $processId -Force -ErrorAction Stop
    $process.WaitForExit(10000) | Out-Null
    Remove-RuntimeState -Path $runtimeStatePath

    $summary = 'BunnyCam stopped successfully.'
    $details = "Stopped process $processId and removed runtime state."
    Add-ResultsEntry -Path $resultsPath -Action 'stop' -Outcome 'success' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $processId -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
    Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
    Write-Host $summary
} catch {
    $summary = 'Failed to stop BunnyCam.'
    $details = $_.Exception.Message
    Add-ResultsEntry -Path $resultsPath -Action 'stop' -Outcome 'failure' -Details $details -Url $rootUrl -Backend $backend -BindHost $bindHost -Port $port -ProcessId $processId -StatusSummary $summary -Actor $Actor -ManagedComponents $managedComponents
    Add-LlsNoteEntry -Path $resultsPath -Actor $Actor -Issue $Issue -Fix $Fix -Note $Note
    throw
}
