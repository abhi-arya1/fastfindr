#!/bin/bash

set -e  # Exit on any error

echo "Setting up project..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "This setup script is designed for macOS. Please install dependencies manually on other platforms."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Homebrew is installed
if ! command_exists brew; then
    echo "Homebrew is not installed. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "Homebrew found"

# Update Homebrew
echo "Updating Homebrew..."
brew update

# Install required dependencies
echo "Installing system dependencies..."

# Core build tools
echo "  • Installing CMake..."
brew install cmake

echo "  • Installing Git (if not already installed)..."
brew install git

# C++ libraries
echo "  • Installing FAISS..."
brew install faiss

echo "  • Installing ONNX Runtime..."
brew install onnxruntime

echo "  • Installing OpenMP..."
brew install libomp

# Python dependencies (for converter)
if command_exists python3; then
    echo "Python 3 found"
else
    echo "  • Installing Python 3..."
    brew install python3
fi

if command_exists pip3; then
    echo "pip3 found"
else
    echo "  • Installing pip3..."
    python3 -m ensurepip --upgrade
fi

# Initialize and update git submodules
echo "🔧 Initializing git submodules..."
git submodule init
git submodule update --recursive

# Update submodules to latest
echo "Updating submodules to latest versions..."
git submodule update --remote --merge

# Set up Python virtual environment for converter
echo "Setting up Python environment for converter..."
cd converter
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# Build the tokenizers-cpp submodule
echo "Building tokenizers-cpp..."
cd third_party/tokenizers-cpp
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
cd ../../..

echo "Creating main build directory..."
if [ ! -d "build" ]; then
    mkdir build
fi

# Build the main project
echo "Building main project..."
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
cd ..

echo ""
echo "Setup complete!"
echo ""
echo "Summary:"
echo "  • All system dependencies installed via Homebrew"
echo "  • Git submodules initialized and updated"
echo "  • Python virtual environment created in converter/venv"
echo "  • tokenizers-cpp submodule built"
echo "  • Main project built successfully"
echo ""
echo "To run the project:"
echo "  ./build/hnsw_search"
echo ""
echo "To use the converter:"
echo "  cd converter"
echo "  source venv/bin/activate"
echo "  python exporter.py"
echo ""
echo "To rebuild after changes:"
echo "  cd build && make -j$(sysctl -n hw.ncpu)"