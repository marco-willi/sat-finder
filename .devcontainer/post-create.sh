#!/bin/bash
# Post-create command: Runs once after container is created

set -e

echo "Running post-create setup..."

# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash


# Install Google Cloud SDK
echo "Installing Google Cloud SDK..."
GCLOUD_DIR="/opt/google-cloud-sdk"
if [ ! -d "$GCLOUD_DIR" ]; then
    curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz -o /tmp/gcloud.tar.gz
    tar -xf /tmp/gcloud.tar.gz -C /opt
    rm /tmp/gcloud.tar.gz
    $GCLOUD_DIR/install.sh --quiet --path-update=false --command-completion=true

    # Install gsutil (usually included, but ensure it's there)
    $GCLOUD_DIR/bin/gcloud components install gsutil --quiet 2>/dev/null || true

    echo "Google Cloud SDK installed."
else
    echo "Google Cloud SDK already installed."
fi

# Add gcloud to PATH globally (works for all shells, not just interactive bash)
GCLOUD_PROFILE="/etc/profile.d/gcloud.sh"
if [ ! -f "$GCLOUD_PROFILE" ]; then
    # shellcheck disable=SC2016
    echo 'export PATH="/opt/google-cloud-sdk/bin:$PATH"' > "$GCLOUD_PROFILE"
    chmod +x "$GCLOUD_PROFILE"
fi

# Also add to bashrc for interactive shells (completion support)
if ! grep -q "google-cloud-sdk" /root/.bashrc 2>/dev/null; then
    # shellcheck disable=SC2016
    echo 'export PATH="/opt/google-cloud-sdk/bin:$PATH"' >> /root/.bashrc
    # shellcheck disable=SC2016
    echo 'source /opt/google-cloud-sdk/completion.bash.inc 2>/dev/null || true' >> /root/.bashrc
fi

# Add to PATH for current session
export PATH="$GCLOUD_DIR/bin:$PATH"
echo "Run 'gcloud auth login' to authenticate with GCP."


# Ensure we're in the workspace
cd /workspace

# Install project dependencies from pyproject.toml
if [ -f "pyproject.toml" ]; then
    echo "Installing dependencies from pyproject.toml..."
    pip install -e ".[dev]"
    echo "Dependencies installed"
else
    echo "Warning: pyproject.toml not found"
fi

# Quick package check
echo "Checking for key packages..."
python -c "import torch; print(f'  torch {torch.__version__}')" 2>/dev/null || echo "  torch not found"
python -c "import gradio; print(f'  gradio installed')" 2>/dev/null || echo "  gradio not found"

# Setup Jupyter kernel
echo ""
echo "Setting up Jupyter kernel..."
python -m ipykernel install --name=sat-finder --display-name="Python (sat-finder)"
echo "Jupyter kernel installed"

# Download OpenSeadragon for Gradio app
echo ""
echo "Setting up static assets for Gradio app..."
mkdir -p static/js static/tiles
if [ ! -f "static/js/openseadragon.min.js" ]; then
    echo "Downloading OpenSeadragon..."
    curl -sL -o static/js/openseadragon.min.js \
        https://unpkg.com/openseadragon@4.1.0/build/openseadragon/openseadragon.min.js
    echo "OpenSeadragon downloaded"
else
    echo "OpenSeadragon already exists"
fi

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo ""
    echo "Installing pre-commit hooks..."
    pre-commit install || echo "Warning: Failed to install pre-commit hooks (continuing anyway)"
fi

# Setup SSH key for Lambda Labs (persists in Docker volume)
echo ""
echo "Setting up SSH keys..."
SSH_DIR="/root/.ssh"
SSH_KEY="$SSH_DIR/id_ed25519"

chmod 700 "$SSH_DIR"

if [ ! -f "$SSH_KEY" ]; then
    echo "Generating new SSH key for Lambda Labs..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "devcontainer-lambda-labs"
    chmod 600 "$SSH_KEY"
    chmod 644 "$SSH_KEY.pub"
    echo ""
    echo "=============================================="
    echo "NEW SSH KEY GENERATED"
    echo "Add this public key to Lambda Labs:"
    echo "https://cloud.lambdalabs.com/ssh-keys"
    echo "=============================================="
    cat "$SSH_KEY.pub"
    echo "=============================================="
else
    echo "SSH key already exists (persisted from previous build)"
fi

# Ensure known_hosts exists
touch "$SSH_DIR/known_hosts"
chmod 644 "$SSH_DIR/known_hosts"

echo ""
echo "Post-create setup complete!"
