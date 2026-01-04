#!/bin/bash
# Post-start command: Runs every time the container starts

set -e

echo "Running post-start tasks..."

# Display current Python environment
echo ""
echo "Python: $(which python) ($(python --version))"

# Check GPU availability
if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || true
    python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || true
fi

echo ""
echo "Container ready!"
echo ""
echo "Useful commands:"
echo "  pip install -e '.[dev]'  - Reinstall dependencies"
echo "  make help                - Show make targets"
echo "  jupyter lab              - Start JupyterLab"
