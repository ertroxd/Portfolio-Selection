#!/usr/bin/env bash
# Einmaliges Setup unter macOS / Linux.
#
#   bash setup.sh
#
# Legt .venv mit Python 3.12 an, installiert die gepinnten Pakete und FinRL in
# einer festen Version direkt von GitHub - ohne dessen Abhaengigkeiten (siehe README).
set -euo pipefail

# Fester FinRL-Stand, mit dem alle Ergebnisse in runs/ erzeugt wurden.
FINRL_COMMIT="2334a5fe6d30629157f13c3b0319e1637e15e123"
FINRL_URL="https://github.com/AI4Finance-Foundation/FinRL/archive/${FINRL_COMMIT}.zip"

cd "$(dirname "$0")"

PY=""
for cand in python3.12 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && \
       [ "$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "3.12" ]; then
        PY="$cand"; break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.12 nicht gefunden (z. B. 'brew install python@3.12')." >&2
    exit 1
fi

[ -d .venv ] || { echo "[setup] Lege .venv an ..."; "$PY" -m venv .venv; }
VPY=".venv/bin/python"

echo "[setup] Installiere Pakete aus requirements.txt ..."
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

echo "[setup] Installiere FinRL @ ${FINRL_COMMIT:0:7} (ohne Abhaengigkeiten) ..."
"$VPY" -m pip install --no-deps "$FINRL_URL"

"$VPY" -c "import finrl_shim; from tr_env import TradeRepublicEnv; print('[setup] OK - Umgebung steht.')"
