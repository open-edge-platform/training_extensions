# Add Git Bash to PATH for cygpath utility
$gitPath = "C:\Program Files\Git\usr\bin"
if (Test-Path $gitPath) {
    $env:PATH = "$gitPath;$env:PATH"
} else {
    Write-Warning "Git Bash not found at $gitPath. Make sure Git for Windows is installed."
}

# ---------------------------------------------------------------------------
# Cleanup: stop stale dev processes from previous runs so the new backend/UI
# can bind their ports and we are guaranteed to serve the latest code.
# ---------------------------------------------------------------------------
function Stop-PortOwners {
    param([int[]]$Ports)

    foreach ($port in $Ports) {
        $owners = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $owners) {
            if ($procId -and $procId -ne 0) {
                $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
                $name = if ($proc) { $proc.ProcessName } else { "PID $procId" }
                Write-Host "Stopping stale process on port $port ($name, PID $procId)..."
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host "Cleaning up previous dev processes..."
# Backend listens on 7860, UI dev server on 3000.
Stop-PortOwners -Ports @(7860, 3000)

# Give the OS a moment to release the ports before we rebind them.
Start-Sleep -Seconds 1

# Clear the rsbuild dev build cache so the UI is rebuilt from the current
# sources (guarantees the dev server serves your latest changes).
$uiCache = Join-Path $PSScriptRoot "ui/node_modules/.cache"
if (Test-Path $uiCache) {
    Write-Host "Clearing UI build cache ($uiCache)..."
    Remove-Item -Path $uiCache -Recurse -Force -ErrorAction SilentlyContinue
}

# Start backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
  `$env:PATH = '$gitPath;' + `$env:PATH
  cd `"$PSScriptRoot/backend`"
  just venv --accelerator cpu
  # Initialize database first
  `$env:PYTHONUNBUFFERED = 1
  `$env:PYTHONPATH = '.'
  Write-Host "Initializing database..."
  uv run app/cli.py init-db
  # Only import demo data on a fresh setup. Re-importing on top of existing
  # projects makes the import abort (the project folder already exists), which
  # would crash startup. Skip it when demo projects are already present.
  `$projectsDir = Join-Path (Get-Location) 'data/projects'
  `$hasProjects = (Test-Path `$projectsDir) -and ((Get-ChildItem -Path `$projectsDir -Directory -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
  if (`$hasProjects) {
    Write-Host 'Existing projects found - skipping demo setup.'
    just run-server
  } else {
    just run-server --setup-demo
  }
"@

# Wait a bit for backend to start
Start-Sleep -Seconds 5

# Start UI in new window. Only run `npm install` when dependencies are missing:
# it triggers a network `preinstall` clone that is slow and can overwrite the
# local `packages/` folder. `rsbuild dev` always serves the latest `src/` code.
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
  cd `"$PSScriptRoot/ui`"
  if (-not (Test-Path 'node_modules')) {
    Write-Host 'node_modules missing - running npm install...'
    npm install
  } else {
    Write-Host 'Dependencies present - skipping npm install (delete node_modules to force a reinstall).'
  }
  npm start
"@

Write-Host "Backend started at http://localhost:7860"
Write-Host "UI will start at http://localhost:3000"
Write-Host "Open the UI at http://localhost:3000 (not the LAN/Network URLs printed by rsbuild)."