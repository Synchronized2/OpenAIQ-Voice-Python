$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\pythonw.exe")) {
    throw "Dependencies are not installed. Run .\setup.ps1 first."
}

& .\.venv\Scripts\pythonw.exe .\gui.py @args
exit $LASTEXITCODE
