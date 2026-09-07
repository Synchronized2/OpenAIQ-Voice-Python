param(
    [string]$Python = "python",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $Python -c "import sys, struct; assert sys.platform == 'win32' and (3, 11) <= sys.version_info[:2] <= (3, 12) and struct.calcsize('P') == 8, 'Use Windows 64-bit Python 3.11 or 3.12'"
    if ($LASTEXITCODE -ne 0) { throw "Unsupported Python interpreter: $Python" }
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
}

& $VenvPython -c "import sys, struct; print('Virtual environment:', sys.executable, sys.version); assert (3, 11) <= sys.version_info[:2] <= (3, 12) and struct.calcsize('P') == 8, 'Existing .venv requires 64-bit Python 3.11 or 3.12'"
if ($LASTEXITCODE -ne 0) { throw "Existing .venv is incompatible. Rename it and run setup again." }

$env:PIP_NO_CACHE_DIR = "1"
$env:MODELSCOPE_CACHE = Join-Path $ProjectRoot ".cache\modelscope"
$env:HF_HOME = Join-Path $ProjectRoot ".cache\huggingface"
$env:TORCH_HOME = Join-Path $ProjectRoot ".cache\torch"

& .\.venv\Scripts\python.exe -m pip install --no-cache-dir --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip uninstall --yes sherpa-onnx sherpa-onnx-core
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path "config.py")) {
    Copy-Item -LiteralPath "config.example.py" -Destination "config.py"
    Write-Host "Created config.py from the safe template. Existing configs are never overwritten."
}

if (-not $SkipModels) {
    & $VenvPython .\download_models.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "Setup completed. Edit config.py, then run .\run-ui.ps1 or .\run.ps1."
