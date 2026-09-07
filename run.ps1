$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Dependencies are not installed. Run .\setup.ps1 first."
}

& .\.venv\Scripts\python.exe .\main.py @args
exit $LASTEXITCODE
