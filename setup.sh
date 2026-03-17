#!/usr/bin/env bash
# Creates a virtual environment and installs all dependencies.
# Usage: bash setup.sh

set -e

VENV_DIR="venv"

echo "Creating virtual environment in ./$VENV_DIR …"
python3 -m venv "$VENV_DIR"

echo "Activating and upgrading pip…"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo "Installing dependencies from requirements.txt …"
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run the pipeline:"
echo "  python run_pipeline.py"
echo ""
echo "Or use the interactive interface (after training):"
echo "  python interface.py --demo"
