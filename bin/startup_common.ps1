$ErrorActionPreference = 'Stop'

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

function Get-DependencyVersions {
    param([string]$PythonExe)

    $pythonSnippet = @"
import json
import platform
from importlib import metadata

packages = {
    'Flask': 'flask',
    'NumPy': 'numpy',
    'OpenCV': 'opencv-python',
    'Waitress': 'waitress',
    'Ultralytics': 'ultralytics',
    'face_recognition': 'face-recognition',
}

result = {'Python': platform.python_version()}
for label, package in packages.items():
    try:
        result[label] = metadata.version(package)
    except metadata.PackageNotFoundError:
        result[label] = 'not installed'

print(json.dumps(result))
"@

    $packageVersions = & $PythonExe -c $pythonSnippet
    if (-not $packageVersions) {
        throw 'Unable to determine Python package versions.'
    }

    $versions = $packageVersions | ConvertFrom-Json -AsHashtable

    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            $versions['Docker'] = $dockerVersion.Trim()
        } else {
            $versions['Docker'] = 'not installed'
        }
    } catch {
        $versions['Docker'] = 'not installed'
    }

    return $versions
}

function Get-LifecycleSectionLines {
    param(
        [string]$BindHost,
        [int]$Port,
        [string]$RuntimeStatePath,
        [string]$StdoutLog,
        [string]$StderrLog
    )

    $rootUrl = "http://$BindHost`:$Port/"

    return @(
        '- Start command: .\start_local.ps1',
        '- Stop command: .\stop_local.ps1',
        "- Default local endpoint: $rootUrl",
        '- Managed components today: BunnyCam Python web process only.',
        '- Docker containers, separate workers, and extra servers: none configured in this repository today.',
        "- Runtime state file: $RuntimeStatePath",
        "- Stdout log: $StdoutLog",
        "- Stderr log: $StderrLog"
    )
}

function Get-VersionSectionLines {
    param([hashtable]$Versions)

    return @(
        "- Python: $($Versions['Python']) (required)",
        "- Flask: $($Versions['Flask']) (required)",
        "- NumPy: $($Versions['NumPy']) (required)",
        "- OpenCV: $($Versions['OpenCV']) (required on Windows laptop backend)",
        "- Waitress: $($Versions['Waitress']) (optional production/local WSGI server)",
        "- Ultralytics: $($Versions['Ultralytics']) (optional detection pipeline)",
        "- face_recognition: $($Versions['face_recognition']) (optional identity pipeline)",
        "- Docker: $($Versions['Docker']) (optional; no containers are launched by current scripts)"
    )
}

function New-ResultsTemplate {
    return @(
        '# BunnyCam Startup Results',
        '',
        'This file is maintained by the startup and shutdown scripts.',
        'It records lifecycle commands, required component versions, run outcomes, and LLS issue/fix/note entries.',
        '',
        '## Lifecycle Commands',
        '<!-- STARTUP_COMMANDS_BEGIN -->',
        '<!-- STARTUP_COMMANDS_END -->',
        '',
        '## Required Components And Versions',
        '<!-- STARTUP_VERSIONS_BEGIN -->',
        '<!-- STARTUP_VERSIONS_END -->',
        '',
        '## Run History',
        '<!-- STARTUP_RUN_HISTORY_BEGIN -->',
        '<!-- STARTUP_RUN_HISTORY_END -->',
        '',
        '## LLS Session Notes',
        '<!-- STARTUP_LLS_NOTES_BEGIN -->',
        '<!-- STARTUP_LLS_NOTES_END -->',
        ''
    )
}

function Set-MarkdownSection {
    param(
        [string]$Path,
        [string]$BeginMarker,
        [string]$EndMarker,
        [string[]]$Lines
    )

    $content = Get-Content -Path $Path -Raw -Encoding utf8
    $begin = [regex]::Escape($BeginMarker)
    $end = [regex]::Escape($EndMarker)
    $replacementBody = ($Lines -join "`r`n")
    $pattern = "(?s)($begin\r?\n)(.*?)((?:\r?\n)$end)"
    $updated = [regex]::Replace($content, $pattern, ('$1' + $replacementBody + '$3'))
    Set-Content -Path $Path -Value $updated -Encoding utf8
}

function Add-MarkdownSectionEntry {
    param(
        [string]$Path,
        [string]$EndMarker,
        [string[]]$Lines
    )

    $content = Get-Content -Path $Path -Raw -Encoding utf8
    $entryText = ($Lines -join "`r`n")
    if ($entryText -and -not $entryText.EndsWith("`r`n")) {
        $entryText += "`r`n"
    }
    $updated = $content.Replace($EndMarker, ($entryText + $EndMarker))
    Set-Content -Path $Path -Value $updated -Encoding utf8
}

function Set-ResultsFileSections {
    param(
        [string]$Path,
        [string[]]$LifecycleLines,
        [string[]]$VersionLines
    )

    $legacyHistory = @()
    $hasMarkers = $false

    if (Test-Path $Path) {
        $existingContent = Get-Content -Path $Path -Raw -Encoding utf8
        $hasMarkers = $existingContent.Contains('<!-- STARTUP_COMMANDS_BEGIN -->')
        if (-not $hasMarkers -and $existingContent.Trim()) {
            $legacyHistory = $existingContent.TrimEnd() -split "\r?\n"
        }
    }

    if (-not $hasMarkers) {
        Set-Content -Path $Path -Value (New-ResultsTemplate) -Encoding utf8
        if ($legacyHistory.Count -gt 0) {
            $historyBlock = @('### Legacy History Migration', '') + $legacyHistory + @('')
            Add-MarkdownSectionEntry -Path $Path -EndMarker '<!-- STARTUP_RUN_HISTORY_END -->' -Lines $historyBlock
        }
    }

    Set-MarkdownSection -Path $Path -BeginMarker '<!-- STARTUP_COMMANDS_BEGIN -->' -EndMarker '<!-- STARTUP_COMMANDS_END -->' -Lines $LifecycleLines
    Set-MarkdownSection -Path $Path -BeginMarker '<!-- STARTUP_VERSIONS_BEGIN -->' -EndMarker '<!-- STARTUP_VERSIONS_END -->' -Lines $VersionLines
}

function Add-ResultsEntry {
    param(
        [string]$Path,
        [string]$Action,
        [string]$Outcome,
        [string]$Details,
        [string]$Url,
        [string]$Backend,
        [string]$BindHost,
        [int]$Port,
        [Nullable[int]]$ProcessId,
        [string]$StatusSummary,
        [string]$Actor,
        [string[]]$ManagedComponents
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $hostname = [System.Net.Dns]::GetHostName()
    $pidText = if ($null -ne $ProcessId) { $ProcessId.ToString() } else { 'n/a' }
    $managedText = if ($ManagedComponents -and $ManagedComponents.Count -gt 0) { $ManagedComponents -join '; ' } else { 'none recorded' }

    $entry = @(
        "### $timestamp | $hostname | $Action | $Outcome",
        '',
        "- Timestamp: $timestamp",
        "- Hostname: $hostname",
        "- Actor: $Actor",
        "- Action: $Action",
        "- Backend: $Backend",
        "- Bind Host: $BindHost",
        "- Port: $Port",
        "- URL: $Url",
        "- PID: $pidText",
        "- Summary: $StatusSummary",
        "- Managed Components: $managedText",
        "- Details: $Details",
        ''
    )

    Add-MarkdownSectionEntry -Path $Path -EndMarker '<!-- STARTUP_RUN_HISTORY_END -->' -Lines $entry
}

function Add-LlsNoteEntry {
    param(
        [string]$Path,
        [string]$Actor,
        [string]$Issue,
        [string]$Fix,
        [string]$Note
    )

    if ($Actor -ne 'LLS' -and [string]::IsNullOrWhiteSpace($Issue) -and [string]::IsNullOrWhiteSpace($Fix) -and [string]::IsNullOrWhiteSpace($Note)) {
        return
    }

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $hostname = [System.Net.Dns]::GetHostName()
    $safeIssue = if ([string]::IsNullOrWhiteSpace($Issue)) { 'none recorded' } else { $Issue }
    $safeFix = if ([string]::IsNullOrWhiteSpace($Fix)) { 'none recorded' } else { $Fix }
    $safeNote = if ([string]::IsNullOrWhiteSpace($Note)) { 'none recorded' } else { $Note }

    $entry = @(
        "### $timestamp | $hostname | $Actor",
        '',
        "- Timestamp: $timestamp",
        "- Hostname: $hostname",
        "- Actor: $Actor",
        "- Issue: $safeIssue",
        "- Fix: $safeFix",
        "- Note: $safeNote",
        ''
    )

    Add-MarkdownSectionEntry -Path $Path -EndMarker '<!-- STARTUP_LLS_NOTES_END -->' -Lines $entry
}

function Write-RuntimeState {
    param(
        [string]$Path,
        [int]$ProcessId,
        [string]$Backend,
        [string]$BindHost,
        [int]$Port,
        [string]$Url,
        [string]$Actor,
        [string[]]$ManagedComponents
    )

    $state = [ordered]@{
        process_id = $ProcessId
        backend = $Backend
        bind_host = $BindHost
        port = $Port
        url = $Url
        actor = $Actor
        hostname = [System.Net.Dns]::GetHostName()
        started_at = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
        managed_components = $ManagedComponents
    }

    $state | ConvertTo-Json -Depth 5 | Set-Content -Path $Path -Encoding utf8
}

function Read-RuntimeState {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }

    return Get-Content -Path $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Remove-RuntimeState {
    param([string]$Path)

    if (Test-Path $Path) {
        Remove-Item -Path $Path -Force
    }
}

function Get-BunnyCamProcessById {
    param([int]$ProcessId)

    try {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    } catch {
        return $null
    }

    if (-not $processInfo) {
        return $null
    }

    if ($processInfo.CommandLine -and $processInfo.CommandLine -match 'sec_cam\.py') {
        return [pscustomobject]@{
            ProcessId = [int]$ProcessId
            CommandLine = $processInfo.CommandLine
            Name = $processInfo.Name
        }
    }

    return $null
}

function Get-BunnyCamProcessFromPort {
    param([int]$Port)

    try {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop | Select-Object -First 1
    } catch {
        return $null
    }

    if (-not $listener) {
        return $null
    }

    return Get-BunnyCamProcessById -ProcessId ([int]$listener.OwningProcess)
}
