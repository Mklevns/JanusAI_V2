#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_CMD="python3"
VENV_DIR="venv-test"
REQUIREMENTS_FILE="requirements-test.txt"

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
    print_message "Virtual environment './$VENV_DIR' already exists. Reusing it."
fi

print_message "✅ Virtual environment is ready."

# 3. Create requirements-test.txt
print_message "Creating '$REQUIREMENTS_FILE' with minimal testing dependencies..."
cat > $REQUIREMENTS_FILE << EOL
# Core dependencies for PPO implementation
torch
gymnasium
sympy
pyyaml

# Testing framework
pytest
EOL

print_message "✅ '$REQUIREMENTS_FILE' created successfully."

# 4. Install Dependencies
print_message "Installing dependencies from '$REQUIREMENTS_FILE'..."
# Activate the venv and install packages
source "$VENV_DIR/bin/activate"
pip install -r $REQUIREMENTS_FILE

print_message "✅ Minimal dependencies installed successfully."

# 5. Final Instructions
echo ""
print_message "--------------------------------------------------"
print_message "Minimal testing environment setup is complete!"
print_message "To activate the virtual environment, run:"
print_message "source $VENV_DIR/bin/activate"
print_message ""
print_message "To run the specified tests, use:"
print_message "pytest test/test_async_ppo.py test/test_ppo_agent.py test/test_ppo_integration.py"
print_message "--------------------------------------------------"
