# Einmaliges Setup unter Windows.
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Legt .venv mit Python 3.12 an, installiert die gepinnten Pakete und FinRL in
# einer festen Version direkt von GitHub - ohne dessen Abhaengigkeiten (siehe README).

$ErrorActionPreference = "Stop"

# Fester FinRL-Stand, mit dem alle Ergebnisse in runs/ erzeugt wurden.
$FINRL_COMMIT = "2334a5fe6d30629157f13c3b0319e1637e15e123"
$FINRL_URL = "https://github.com/AI4Finance-Foundation/FinRL/archive/$FINRL_COMMIT.zip"

Set-Location $PSScriptRoot

function Test-Py312([string]$exe, [string[]]$pre) {
    # try/catch statt stderr-Umleitung: unter Windows PowerShell 5.1 wird stderr
    # eines nativen Programms bei ErrorActionPreference=Stop sonst zur Exception.
    try {
        $v = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])"
        return ($LASTEXITCODE -eq 0 -and "$v".Trim() -eq "3.12")
    } catch { return $false }
}

# --- Python 3.12 finden ------------------------------------------------------
$pyExe = $null; $pyPre = @()
if ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-Py312 "py" @("-3.12"))) {
    $pyExe = "py"; $pyPre = @("-3.12")
} elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-Py312 "python" @())) {
    $pyExe = "python"
}
if (-not $pyExe) {
    throw "Python 3.12 nicht gefunden. Installieren von https://www.python.org/downloads/ (3.12.x), dann erneut starten."
}

# --- venv + Pakete -----------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "[setup] Lege .venv an ..."
    & $pyExe @pyPre -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv konnte nicht angelegt werden." }
}
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "[setup] Installiere Pakete aus requirements.txt ..."
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt fehlgeschlagen." }

Write-Host "[setup] Installiere FinRL @ $($FINRL_COMMIT.Substring(0,7)) (ohne Abhaengigkeiten) ..."
& $vpy -m pip install --no-deps $FINRL_URL
if ($LASTEXITCODE -ne 0) { throw "FinRL-Installation fehlgeschlagen." }

# --- Pruefen -----------------------------------------------------------------
& $vpy -c "import finrl_shim; from tr_env import TradeRepublicEnv; print('[setup] OK - Umgebung steht.')"
if ($LASTEXITCODE -ne 0) { throw "Import-Test fehlgeschlagen." }
