# Windows setup (PowerShell). Run from the project folder:  .\setup.ps1
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process Bypass
$ErrorActionPreference = "Stop"

$py = $null
foreach ($cand in @("py -3.12", "py -3.11", "py -3.13", "python")) {
    try {
        $v = & cmd /c "$cand --version 2>&1"
        if ($v -match "Python 3\.(1[1-9])") { $py = $cand; break }
    } catch { }
}
if (-not $py) { throw "Python 3.11+ not found. Install it from python.org (tick 'Add to PATH')." }
Write-Host "Using: $py"

& cmd /c "$py -m venv .venv"
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e . --no-deps

Write-Host ""
Write-Host "Done. Try:"
Write-Host "  .\.venv\Scripts\python.exe -m georeg demo"
Write-Host "  .\run_tests.ps1"
Write-Host "Optional GDAL bindings: see README, section 'Windows'."
