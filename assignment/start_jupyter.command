#!/bin/bash
# ArtExtract — Jupyter Lab Launcher
# Double-click this file in Finder to start

cd "$(dirname "$0")"

echo "Activating .venv_new (Python 3.14)..."
source .venv_new/bin/activate

echo "All packages already installed:"
echo "  torch $(python -c 'import torch; print(torch.__version__)')"
echo "  jupyterlab $(python -c 'import jupyterlab; print(jupyterlab.__version__)')"
echo "  numpy $(python -c 'import numpy; print(numpy.__version__)')"
echo ""
echo "Launching Jupyter Lab at http://localhost:8888"
echo "Press Ctrl+C to stop."
echo ""

pkill -f "jupyter" 2>/dev/null; sleep 1
python -m jupyterlab --port=8888 --no-browser --ServerApp.token='' --ServerApp.password=''
