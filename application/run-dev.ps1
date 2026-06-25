# Add Git Bash to PATH for cygpath utility
$gitPath = "C:\Program Files\Git\usr\bin"
if (Test-Path $gitPath) {
    $env:PATH = "$gitPath;$env:PATH"
} else {
    Write-Warning "Git Bash not found at $gitPath. Make sure Git for Windows is installed."
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

# Start UI in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$PSScriptRoot/ui`"; npm install; npm start"

Write-Host "Backend started at http://localhost:7860"
Write-Host "UI will start at http://localhost:3000"