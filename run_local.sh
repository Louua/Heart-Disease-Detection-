#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "🩺 CARDIOCHECK AI - LOCAL STARTUP SCRIPT"
echo "=================================================="

# 1. Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install it first."
    exit 1
fi

# 2. Check if virtual environment can be created
USE_VENV=true
if [ ! -d "venv" ]; then
    echo "📦 Attempting to create python virtual environment 'venv'..."
    if python3 -m venv venv 2>/dev/null; then
        echo "✅ Virtual environment 'venv' created successfully."
    else
        echo "⚠️  Could not create virtual environment (ensurepip/python3-venv missing)."
        echo "👉 Falling back to user-level package installation (~/.local/)."
        USE_VENV=false
    fi
else
    echo "✅ Virtual environment 'venv' already exists."
fi

# 3. Activate venv or set up pip commands
PIP_CMD="pip"
PIP_FLAGS=""

if [ "$USE_VENV" = true ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
else
    # Install pip for user if not exists
    if ! python3 -m pip --version &>/dev/null; then
        echo "📥 Pip not found. Downloading and installing pip for user..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py --user --break-system-packages
        rm get-pip.py
    fi
    PIP_CMD="python3 -m pip"
    PIP_FLAGS="--user --break-system-packages"
fi

# 4. Install dependencies
echo "📥 Installing requirements..."
$PIP_CMD install $PIP_FLAGS -r requirements.txt

# 5. Check if model pickles exist, if not, train it
if [ ! -f "app/deployment_model.pkl" ] || [ ! -f "app/deployment_preprocessor.pkl" ]; then
    echo "🧠 Model artifacts not found. Starting dataset download and training pipeline..."
    python3 train_local.py
else
    echo "🧠 Found existing model artifacts in app/ folder."
    # Ask if user wants to retrain
    read -t 5 -p "Do you want to retrain the model? [y/N] (Auto-skipping in 5s) " answer || answer="n"
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "🧠 Retraining model..."
        python3 train_local.py
    fi
fi

# 6. Run Streamlit App
echo "🚀 Launching Streamlit App..."
if [ "$USE_VENV" = true ]; then
    streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
else
    # Add ~/.local/bin to PATH in case streamlit is installed there
    export PATH="$HOME/.local/bin:$PATH"
    python3 -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
fi
