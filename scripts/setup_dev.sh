#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_CMD="python3"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements-dev.txt"

# --- Functions ---

# Function to print colored messages
print_message() {
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color
    echo -e "${GREEN}$1${NC}"
}

# --- Main Script ---

# 1. Check for Python
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

print_message "✅ Python check passed."

# 2. Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    print_message "Creating virtual environment in './$VENV_DIR'..."
    $PYTHON_CMD -m venv $VENV_DIR
else
    print_message "Virtual environment './$VENV_DIR' already exists. Skipping creation."
fi

print_message "✅ Virtual environment is ready."

# 3. Create requirements-dev.txt
print_message "Creating '$REQUIREMENTS_FILE' with all dependencies..."
cat > $REQUIREMENTS_FILE << EOL
# Core Dependencies
torch
gymnasium
sympy
pyyaml
wandb
typer

# Discovered Dependencies
pydantic
matplotlib
pandas
ray[tune]
stable-baselines3

# Development & Testing Tools
pytest
pytest-cov
streamlit
EOL

print_message "✅ '$REQUIREMENTS_FILE' created successfully."

# 4. Install Dependencies
print_message "Installing dependencies from '$REQUIREMENTS_FILE'..."
# Activate the venv and install packages in a subshell to keep the current shell clean
(
    source "$VENV_DIR/bin/activate"
    pip install -r $REQUIREMENTS_FILE
)

print_message "✅ Dependencies installed successfully."

# 5. Final Instructions
echo ""
print_message "--------------------------------------------------"
print_message "Development environment setup is complete!"
print_message "To activate the virtual environment, run:"
print_message "source $VENV_DIR/bin/activate"
print_message "--------------------------------------------------"