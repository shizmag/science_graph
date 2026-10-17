[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('doctor', 'install', 'smoke', 'run', 'resume', 'status', 'pack', 'verify')]
    [string]$Command,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Remaining
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Version = '1.1.0'
$Image = "biomedical-rag-cuda:$Version"
$BaseImageDigest = 'sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2'
$State = if ($env:ABLA_STATE_DIR) { $env:ABLA_STATE_DIR } else { Join-Path $env:LOCALAPPDATA 'ablation_cuda' }
$Models = Join-Path $State 'models'
$Database = Join-Path $State 'database'
$Logs = Join-Path $State 'logs'
$Outputs = Join-Path $State 'outputs'
$ReturnDir = if ($env:ABLA_RETURN_DIR) { $env:ABLA_RETURN_DIR } else { Join-Path $Root 'return' }
New-Item -ItemType Directory -Force -Path $Models, $Database, $Logs, $Outputs, $ReturnDir | Out-Null

function Fail([string]$Message) {
    throw "ERROR: $Message"
}

function Invoke-Docker([string[]]$Arguments) {
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "Docker command failed with exit code $LASTEXITCODE." }
}

function Test-HostPrerequisites {
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        Fail "This CUDA runtime is built for linux/amd64. Use 64-bit x86 Windows 11 with an RTX 5090; detected '$env:PROCESSOR_ARCHITECTURE'."
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Fail 'Docker Desktop is missing. Install Docker Desktop with its WSL2 backend, then rerun .\ablation.ps1 doctor.'
    }
    $OsType = (& docker info --format '{{.OSType}}' 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or $OsType -ne 'linux') {
        Fail "Docker Desktop must be in Linux-container mode (docker info reported '$OsType'). Switch to Linux containers, then rerun."
    }
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        Fail 'WSL2 is unavailable. Enable/install WSL2 using your approved Windows administration process; this script makes no system changes.'
    }
    $WslStatus = (& wsl.exe --status 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { Fail "WSL2 status check failed: $WslStatus" }
    $WslList = (& wsl.exe --list --verbose 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or $WslList -notmatch '(?m)\s2\s*$') {
        Fail 'No WSL2 distribution was detected. Configure a WSL2 distribution and Docker Desktop WSL2 integration before running the CUDA smoke.'
    }
    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        Fail 'The NVIDIA Windows WSL driver is not visible. Install an NVIDIA Windows driver with WSL2 CUDA support (580.88 or newer for CUDA 13 / Blackwell) using your normal driver process; this script will not replace it.'
    }
    $DriverLine = (& nvidia-smi.exe --query-gpu=name,driver_version --format=csv,noheader 2>&1 | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0) { Fail "NVIDIA Windows driver check failed: $DriverLine" }
    $DriverMatch = [regex]::Match($DriverLine, '(\d+\.\d+(?:\.\d+)?)\s*$')
    if (-not $DriverMatch.Success -or [version]$DriverMatch.Groups[1].Value -lt [version]'580.88') {
        Fail "NVIDIA Windows/WSL driver must be version 580.88 or newer for the pinned CUDA 13 runtime. Detected: $DriverLine"
    }
    $WslGpu = (& wsl.exe -e nvidia-smi 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        Fail "NVIDIA GPU passthrough is unavailable in WSL2. Update the Windows NVIDIA driver and enable Docker Desktop WSL2 GPU support; no automatic driver or administrative changes were attempted. Details: $WslGpu"
    }
}

function Ensure-Image {
    & docker buildx version *> $null
    if ($LASTEXITCODE -ne 0) { Fail 'Docker Buildx is required to record an immutable OCI image digest. Update Docker Engine/Docker Desktop.' }
    $MetadataFile = Join-Path $State 'image-metadata.json'
    & docker image inspect $Image *> $null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $MetadataFile)) {
        Write-Host 'Building the pinned linux/amd64 benchmark image (first build downloads locked runtime wheels).'
        Invoke-Docker @('buildx', 'build', '--platform', 'linux/amd64', '--pull', '--load',
            '--metadata-file', $MetadataFile, '-t', $Image, '-f', (Join-Path $Root 'Dockerfile'), $Root)
    }
    $script:ImageId = (& docker image inspect --format '{{.Id}}' $Image | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or $script:ImageId -notmatch '^sha256:') { Fail 'Docker did not return an immutable local image ID.' }
    $ImageMetadata = Get-Content $MetadataFile -Raw | ConvertFrom-Json
    $script:ImageDigest = $ImageMetadata.'containerimage.digest'
    if ($script:ImageDigest -notmatch '^sha256:') { Fail 'Docker Buildx did not return an OCI image digest.' }
}

function Invoke-Container([string]$RuntimeCommand, [string[]]$Extra = @(), [string[]]$Mounts = @()) {
    $GpuArgs = @()
    if ($RuntimeCommand -in @('doctor', 'install', 'smoke', 'run', 'resume')) { $GpuArgs = @('--gpus', 'all') }
    $Args = @('run', '--rm', '--platform', 'linux/amd64') + $GpuArgs + @(
        '-v', "${Root}:/app/release:ro",
        '-v', "${Models}:/models", '-v', "${Database}:/db", '-v', "${Logs}:/logs",
        '-v', "${Outputs}:/outputs", '-v', "${ReturnDir}:/return"
    )
    foreach ($Mount in $Mounts) { $Args += @('--mount', $Mount) }
    $Args += @(
        '-e', "ABLA_IMAGE_ID=$script:ImageId", '-e', "ABLA_IMAGE_DIGEST=$script:ImageDigest",
        '-e', "ABLA_BASE_IMAGE_DIGEST=$BaseImageDigest", '-e', 'ABLA_UV_VERSION=0.10.9',
        '-e', "ABLA_HOST_OUTPUT_ROOT=$Outputs", '-e', "ABLA_HOST_RETURN_DIR=$ReturnDir",
        '-e', 'HOME=/state/home', $Image, $RuntimeCommand
    ) + $Extra
    Invoke-Docker $Args
}

switch ($Command) {
    'doctor' {
        Test-HostPrerequisites
        Ensure-Image
        try { Invoke-Container 'doctor' }
        catch { Fail 'NVIDIA GPU passthrough is not visible inside Docker Desktop. Update the compatible Windows WSL driver and enable Docker Desktop WSL2 GPU integration; no driver or system setting was changed.' }
    }
    'install' {
        Test-HostPrerequisites
        Ensure-Image
        Invoke-Container 'install'
    }
    'smoke' {
        Test-HostPrerequisites
        Ensure-Image
        try { Invoke-Container 'doctor' }
        catch { Fail 'CUDA operations are unavailable inside the container. Resolve the host NVIDIA driver/WSL2 GPU passthrough prerequisite before downloading model assets.' }
        Invoke-Container 'smoke'
    }
    'run' {
        Test-HostPrerequisites
        Ensure-Image
        Invoke-Container 'run' $Remaining
    }
    'resume' {
        Test-HostPrerequisites
        Ensure-Image
        if ($Remaining.Count -gt 1) { Fail 'Usage: .\ablation.ps1 resume [run_directory_name]' }
        if ($Remaining.Count -eq 1) {
            Invoke-Container 'resume' @('--run-id', $Remaining[0])
        } else {
            Invoke-Container 'resume'
        }
    }
    'status' {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail 'Docker Desktop is missing.' }
        Ensure-Image
        Invoke-Container 'status'
    }
    'pack' {
        Ensure-Image
        Invoke-Container 'pack' (@('--export-dir', '/return') + $Remaining)
    }
    'verify' {
        if ($Remaining.Count -lt 1) { Fail 'Usage: .\ablation.ps1 verify <results.tar.gz>' }
        Ensure-Image
        $Archive = (Resolve-Path -LiteralPath $Remaining[0]).Path
        $ArchiveName = [System.IO.Path]::GetFileName($Archive)
        $ArchiveId = $ArchiveName -replace '\.tar\.gz$', ''
        $Mounts = @("type=bind,source=$Archive,target=/verify/archive.tar.gz,readonly")
        if (Test-Path -LiteralPath "$Archive.sha256") {
            $Mounts += "type=bind,source=$Archive.sha256,target=/verify/archive.tar.gz.sha256,readonly"
        }
        Invoke-Container -RuntimeCommand 'verify' `
            -Extra @('--archive', '/verify/archive.tar.gz', '--destination', "/return/imported/$ArchiveId") `
            -Mounts $Mounts
    }
}
