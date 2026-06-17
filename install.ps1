# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    Install Intel Geti application and its dependencies on Windows.

.DESCRIPTION
    This script installs the Intel Geti application, including uv (Python package manager),
    Node.js/npm, and builds both the backend and frontend.

.PARAMETER Verbose
    Show detailed output from all commands.

.PARAMETER Yes
    Assume yes to all prompts (non-interactive mode).

.PARAMETER WorkDir
    Set the working directory (default: $env:USERPROFILE\geti).

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Verbose -Yes
    .\install.ps1 -WorkDir "C:\my\custom\path"
#>

[CmdletBinding()]
param(
    [Alias("y")]
    [switch]$Yes,

    [Alias("w")]
    [string]$WorkDir = "$env:USERPROFILE\geti"
)

$ErrorActionPreference = "Stop"

$GIT_URL = "https://github.com/open-edge-platform/training_extensions.git"
$GIT_BRANCH = "develop"

$BUILD_TOOLS_DIR = Join-Path $WorkDir ".build"
$UV_DIR = Join-Path $BUILD_TOOLS_DIR "uv"
$NVM_DIR = Join-Path $BUILD_TOOLS_DIR "nvm"
$LOG_FILE = Join-Path $BUILD_TOOLS_DIR ".install.log"

$script:NPM_BIN = ""

function Write-Step {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-ErrorMessage {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

function Confirm-Prompt {
    param([string]$Prompt)

    if ($Yes) { return $true }

    $response = Read-Host "$Prompt [Y/n]"
    if ($response -match "^n(o)?$") { return $false }
    return $true
}

function Invoke-Cmd {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    # Temporarily allow stderr output without terminating (tools like npm/git
    # write warnings to stderr even on success).
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        if ($VerbosePreference -eq "Continue") {
            & $Command @Arguments 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    Write-Host $_.ToString() -ForegroundColor Yellow
                } else {
                    Write-Host $_
                }
            }
        } else {
            & $Command @Arguments *>> $LOG_FILE
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }

    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Command '$Command $($Arguments -join ' ')' failed with exit code $LASTEXITCODE"
    }
}

function Get-RequiredUvVersion {
    $pyprojectPath = Join-Path $WorkDir "application\backend\pyproject.toml"
    $content = Get-Content $pyprojectPath -Raw

    if ($content -match '\[tool\.uv\][\s\S]*?required-version\s*=\s*"[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)') {
        return $Matches[1]
    }

    throw "Could not parse uv version from pyproject.toml"
}

function Get-RequiredNodeVersion {
    $packageJsonPath = Join-Path $WorkDir "application\ui\package.json"
    $json = Get-Content $packageJsonPath -Raw | ConvertFrom-Json

    $nodeConstraint = $json.engines.node
    if ($nodeConstraint -match '>=v?([0-9]+\.[0-9]+\.[0-9]+)') {
        return $Matches[1]
    }

    throw "Could not parse node version from package.json"
}

function Get-RequiredNpmVersion {
    $packageJsonPath = Join-Path $WorkDir "application\ui\package.json"
    $json = Get-Content $packageJsonPath -Raw | ConvertFrom-Json

    $npmConstraint = $json.engines.npm
    if ($npmConstraint -match '>=([0-9]+\.[0-9]+\.[0-9]+)') {
        return $Matches[1]
    }

    throw "Could not parse npm version from package.json"
}

function Install-Uv {
    $uvVersion = Get-RequiredUvVersion
    $uvExe = Join-Path $UV_DIR "uv.exe"

    if (Test-Path $uvExe) {
        $installedVersion = & $uvExe --version | ForEach-Object { ($_ -split ' ')[1] }
        if ($installedVersion -eq $uvVersion) {
            Write-Step "uv $uvVersion found in $UV_DIR"
            return
        } else {
            Write-Step "uv version mismatch: installed=$installedVersion, required=$uvVersion. Reinstalling..."
        }
    }

    Write-Step "Installing uv $uvVersion to: $UV_DIR"
    if (-not (Confirm-Prompt "Would you like to install uv now?")) {
        Write-ErrorMessage "uv installation skipped. Cannot continue without uv."
        exit 1
    }

    if (-not (Test-Path $UV_DIR)) {
        New-Item -ItemType Directory -Path $UV_DIR -Force | Out-Null
    }

    $installerUrl = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-installer.ps1"
    $env:UV_INSTALL_DIR = $UV_DIR

    if ($VerbosePreference -eq "Continue") {
        & powershell -ExecutionPolicy Bypass -Command "irm '$installerUrl' | iex"
    } else {
        & powershell -ExecutionPolicy Bypass -Command "irm '$installerUrl' | iex" *>> $LOG_FILE
    }

    Remove-Item Env:\UV_INSTALL_DIR -ErrorAction SilentlyContinue
    Write-Step "uv installation complete."
}

function Install-Nvm {
    $nvmExe = Join-Path $NVM_DIR "nvm.exe"

    if (Test-Path $nvmExe) {
        Write-Step "nvm found in $NVM_DIR."
        return
    }

    Write-Step "Installing nvm-windows to: $NVM_DIR"
    if (-not (Confirm-Prompt "Would you like to install nvm-windows now?")) {
        Write-ErrorMessage "nvm installation skipped. Cannot continue without nvm."
        exit 1
    }

    if (-not (Test-Path $NVM_DIR)) {
        New-Item -ItemType Directory -Path $NVM_DIR -Force | Out-Null
    }

    # Download nvm-windows noinstall zip
    $nvmVersion = "1.2.2"
    $nvmZipUrl = "https://github.com/coreybutler/nvm-windows/releases/download/$nvmVersion/nvm-noinstall.zip"
    $nvmZipPath = Join-Path $BUILD_TOOLS_DIR "nvm-noinstall.zip"

    Write-Host "Downloading nvm-windows $nvmVersion..."
    Invoke-WebRequest -Uri $nvmZipUrl -OutFile $nvmZipPath -UseBasicParsing

    Expand-Archive -Path $nvmZipPath -DestinationPath $NVM_DIR -Force
    Remove-Item $nvmZipPath -Force

    # Configure nvm settings
    $nodeDir = Join-Path $NVM_DIR "nodejs"
    $settingsContent = @"
root: $NVM_DIR
path: $nodeDir
"@
    Set-Content -Path (Join-Path $NVM_DIR "settings.txt") -Value $settingsContent

    Write-Step "nvm-windows installation complete."
}

function Install-Npm {
    $requiredNodeVersion = Get-RequiredNodeVersion
    $requiredNpmVersion = Get-RequiredNpmVersion
    $nvmExe = Join-Path $NVM_DIR "nvm.exe"
    $nodeDir = Join-Path $NVM_DIR "nodejs"
    $nodeVersionDir = Join-Path $NVM_DIR "v$requiredNodeVersion"

    # Check if the required node version is already installed
    $nodeBin = Join-Path $nodeVersionDir "node.exe"
    $npmBin = Join-Path $nodeVersionDir "npm.cmd"

    if (Test-Path $nodeBin) {
        $script:NPM_BIN = $npmBin
        $env:PATH = "$nodeVersionDir;$env:PATH"
        $installedNpmVersion = & $npmBin --version 2>$null
        if ($installedNpmVersion -and ([version]$installedNpmVersion -ge [version]$requiredNpmVersion)) {
            Write-Step "node $requiredNodeVersion and npm $installedNpmVersion found."
            return
        }

        Write-Step "npm version too old: installed=$installedNpmVersion, required>=$requiredNpmVersion. Upgrading..."
        Invoke-Cmd -Command $npmBin -Arguments @("install", "-g", "npm@$requiredNpmVersion")
        return
    }

    Write-Step "Required node $requiredNodeVersion not found. Installing..."

    if (-not (Confirm-Prompt "Would you like to install node/npm now?")) {
        Write-ErrorMessage "node/npm installation skipped. Cannot continue without node/npm."
        exit 1
    }

    # Set NVM_HOME for nvm.exe to work properly
    $env:NVM_HOME = $NVM_DIR
    $env:NVM_SYMLINK = $nodeDir

    # Install node (nvm install does not require elevation)
    Invoke-Cmd -Command $nvmExe -Arguments @("install", $requiredNodeVersion)

    # Skip "nvm use" as it requires admin elevation to create a symlink.
    # Instead, we reference binaries directly from the version-specific directory
    # and prepend to PATH so node/npm can find each other.
    $env:PATH = "$nodeVersionDir;$env:PATH"

    $script:NPM_BIN = $npmBin

    if (Test-Path $npmBin) {
        $installedNpmVersion = & $npmBin --version 2>$null
        if ($installedNpmVersion -and ([version]$installedNpmVersion -lt [version]$requiredNpmVersion)) {
            Invoke-Cmd -Command $npmBin -Arguments @("install", "-g", "npm@$requiredNpmVersion")
        }
    }

    Write-Step "node/npm installation complete."
}

function Find-NvidiaGpus {
    $gpuCount = 0

    # Try nvidia-smi
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            $gpus = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($gpus) {
                $gpuCount = ($gpus | Measure-Object -Line).Lines
                if ($gpuCount -gt 0) {
                    Write-Step "Detected $gpuCount NVIDIA GPU(s) via nvidia-smi:"
                    & nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
                    return $true
                }
            }
        } catch {}
    }

    # Try WMI/CIM
    try {
        $gpus = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
        if ($gpus) {
            $gpuCount = @($gpus).Count
            Write-Step "Detected $gpuCount NVIDIA GPU(s):"
            $gpus | ForEach-Object { Write-Host "  $($_.Name)" }
            return $true
        }
    } catch {}

    Write-Host "No NVIDIA GPUs detected."
    return $false
}

function Find-IntelGpus {
    # Try WMI/CIM
    try {
        $gpus = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "Intel" -and $_.Name -match "Arc|Iris|UHD|HD Graphics" }
        if ($gpus) {
            $gpuCount = @($gpus).Count
            Write-Step "Detected $gpuCount Intel GPU(s):"
            $gpus | ForEach-Object { Write-Host "  $($_.Name)" }
            return $true
        }
    } catch {}

    Write-Host "No Intel GPUs detected."
    return $false
}

function Test-PreflightChecks {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git is not installed. Please install git and try again."
    }
}

function Invoke-EnsureSourceCode {
    # Git commands write informational messages to stderr which PowerShell
    # treats as terminating errors under $ErrorActionPreference = "Stop".
    # We temporarily switch to Continue for all git invocations here.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        if (-not (Test-Path $WorkDir)) {
            Write-Step "Cloning Intel Geti repository from $GIT_URL..."
            & git -c advice.detachedHead=false clone --branch $GIT_BRANCH $GIT_URL $WorkDir 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit code $LASTEXITCODE)" }
        } else {
            Write-Step "Work directory $WorkDir already exists, skipping clone."

            $remoteUrl = (& git -C $WorkDir remote get-url origin 2>$null)
            if ($remoteUrl -ne $GIT_URL) {
                throw "$WorkDir remote origin is '$remoteUrl', expected '$GIT_URL'. Remove $WorkDir and re-run the installer."
            }

            $currentSha = (& git -C $WorkDir rev-parse HEAD 2>$null)
            & git -C $WorkDir fetch origin $GIT_BRANCH --tags 2>&1 | Out-Null
            $expectedSha = (& git -C $WorkDir rev-parse "origin/$GIT_BRANCH" 2>$null)
            if (-not $expectedSha) {
                $expectedSha = (& git -C $WorkDir rev-parse $GIT_BRANCH 2>$null)
            }

            if ($currentSha -ne $expectedSha) {
                Write-Step "Switching to $GIT_BRANCH..."
                & git -c advice.detachedHead=false -C $WorkDir checkout $GIT_BRANCH 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "git checkout failed (exit code $LASTEXITCODE)" }
            }
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }
}

function Install-BuildTools {
    Install-Uv
    Install-Nvm
    Install-Npm
}

function Find-Hardware {
    $script:HAS_NVIDIA_GPU = $false
    $script:HAS_INTEL_GPU = $false

    if (Find-NvidiaGpus) {
        $script:HAS_NVIDIA_GPU = $true
    }

    if (Find-IntelGpus) {
        $script:HAS_INTEL_GPU = $true
    }

    if ($script:HAS_NVIDIA_GPU) {
        $script:ACCELERATOR = "cuda"
    } elseif ($script:HAS_INTEL_GPU) {
        $script:ACCELERATOR = "xpu"
    } else {
        $script:ACCELERATOR = "cpu"
    }

    $env:ACCELERATOR = $script:ACCELERATOR
}

function Build-Backend {
    Write-Step "Building venv using accelerator: $($script:ACCELERATOR)"
    $backendDir = Join-Path $WorkDir "application\backend"
    Push-Location $backendDir

    try {
        $uvExe = Join-Path $UV_DIR "uv.exe"

        if ($VerbosePreference -eq "Continue") {
            & $uvExe sync --frozen --extra mqtt --extra $script:ACCELERATOR
        } else {
            & $uvExe sync --frozen --extra mqtt --extra $script:ACCELERATOR --quiet
        }

        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

        Write-Step "Generating OpenAPI specification..."
        $env:PYTHONPATH = "."
        & $uvExe run --no-sync app/cli.py gen-api --target-path openapi.json
        if ($LASTEXITCODE -ne 0) { throw "OpenAPI generation failed" }

        $uiApiDir = Join-Path $WorkDir "application\ui\src\api"
        Copy-Item -Path "openapi.json" -Destination (Join-Path $uiApiDir "openapi-spec.json") -Force
    } finally {
        Pop-Location
    }
}

function Build-Frontend {
    $uiDir = Join-Path $WorkDir "application\ui"
    Push-Location $uiDir

    try {
        Write-Step "Installing UI dependencies with npm..."
        $env:npm_config_yes = "true"
        Invoke-Cmd -Command $script:NPM_BIN -Arguments @("ci")

        Write-Step "Building API client with npm..."
        Invoke-Cmd -Command $script:NPM_BIN -Arguments @("run", "build:api")

        Write-Step "Building UI with npm..."
        $env:ASSET_PREFIX = "/html"
        Invoke-Cmd -Command $script:NPM_BIN -Arguments @("run", "build")
        Remove-Item Env:\ASSET_PREFIX -ErrorAction SilentlyContinue
    } finally {
        Pop-Location
    }
}

function Deploy-Frontend {
    $htmlDir = Join-Path $WorkDir "application\backend\html"

    Write-Step "Copying built UI to backend html directory..."
    if (Test-Path $htmlDir) {
        Remove-Item -Path $htmlDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $htmlDir -Force | Out-Null

    $distDir = Join-Path $WorkDir "application\ui\dist\*"
    Copy-Item -Path $distDir -Destination $htmlDir -Recurse -Force
}

function Register-ShellCommand {
    $uvExe = Join-Path $UV_DIR "uv.exe"
    $backendDir = Join-Path $WorkDir "application\backend"

    # Create a geti.cmd batch file in the work directory
    $cmdPath = Join-Path $WorkDir "geti.cmd"
    $cmdContent = @"
@echo off
pushd "$backendDir"
set STATIC_FILES_DIR=html
"$uvExe" run app/main.py %*
popd
"@
    Set-Content -Path $cmdPath -Value $cmdContent

    # Create a geti.ps1 PowerShell wrapper
    $ps1Path = Join-Path $WorkDir "geti.ps1"
    $ps1Content = @"
# Intel Geti launcher
param([Parameter(ValueFromRemainingArguments=`$true)]`$Args)
Push-Location "$backendDir"
try {
    `$env:STATIC_FILES_DIR = "html"
    & "$uvExe" run app/main.py @Args
} finally {
    Pop-Location
}
"@
    Set-Content -Path $ps1Path -Value $ps1Content

    # Add to PowerShell profile
    $profileDir = Split-Path $PROFILE -Parent
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }
    if (-not (Test-Path $PROFILE)) {
        New-Item -ItemType File -Path $PROFILE -Force | Out-Null
    }

    $beginMarker = "# BEGIN Intel Geti"
    $endMarker = "# END Intel Geti"
    $profileContent = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue

    # Remove old marker block if present
    if ($profileContent -and $profileContent -match [regex]::Escape($beginMarker)) {
        $profileContent = $profileContent -replace "(?s)\r?\n?$([regex]::Escape($beginMarker)).*?$([regex]::Escape($endMarker))\r?\n?", ""
        Set-Content -Path $PROFILE -Value $profileContent -NoNewline
    }

    $functionBlock = @"

$beginMarker
function geti { Push-Location "$backendDir"; try { `$env:STATIC_FILES_DIR = "html"; & "$uvExe" run app/main.py @args } finally { Pop-Location } }
$endMarker
"@
    Add-Content -Path $PROFILE -Value $functionBlock

    Write-Step "Function 'geti' written to $PROFILE"
    Write-Host "Run '. `$PROFILE' to activate it in the current session."
    Write-Host "Example: `$env:HOST='0.0.0.0'; `$env:PORT='8080'; geti"
    Write-Host ""
    Write-Host "Batch file also available at: $cmdPath"
}

# ─── Main ────────────────────────────────────────────────────────────────────

function Main {
    Write-Host ""
    Write-Host "Intel Geti Installer (Windows/PowerShell)" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""

    Test-PreflightChecks
    Invoke-EnsureSourceCode

    # Initialize log file and build tools directory
    if (-not (Test-Path $BUILD_TOOLS_DIR)) {
        New-Item -ItemType Directory -Path $BUILD_TOOLS_DIR -Force | Out-Null
    }
    "" | Set-Content -Path $LOG_FILE

    Install-BuildTools
    Find-Hardware
    Build-Backend
    Build-Frontend
    Deploy-Frontend
    Register-ShellCommand

    Write-Host ""
    Write-Step "Installation complete!"
    Write-Host ""
}

try {
    Main
} catch {
    Write-Host ""
    Write-ErrorMessage "Installation failed: $_"
    if (Test-Path $LOG_FILE) {
        Write-Host "Check $LOG_FILE for details."
    }
    Write-Host "Re-run with -Verbose for more details."
    exit 1
}

