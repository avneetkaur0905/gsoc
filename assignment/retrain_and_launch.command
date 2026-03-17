#!/bin/bash
# ── Retrain Style models (consistent pipeline) then launch JupyterLab ─────────
cd "$(dirname "$0")"

echo "============================================================"
echo "  Step 1: Activating .venv_new (Python 3.14)"
echo "============================================================"
source .venv_new/bin/activate

echo ""
echo "============================================================"
echo "  Step 2: Retraining Style CNN + CNN+RNN with ImageNet norm"
echo "  (this takes ~5-10 minutes on M1 Pro)"
echo "============================================================"
python retrain_style.py 2>&1 | tee retrain_style_log.txt

echo ""
echo "============================================================"
echo "  Step 3: Launching JupyterLab"
echo "============================================================"
pkill -f "jupyter" 2>/dev/null; sleep 1
jupyter lab --port=8888 --no-browser --ServerApp.token='' --ServerApp.password=''
