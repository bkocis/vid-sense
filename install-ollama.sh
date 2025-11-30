#!/bin/bash

# Ollama Installation Script for macOS
# This script installs Ollama using the recommended method

set -e  # Exit on error

echo "🚀 Starting Ollama installation for macOS..."

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed!"
    ollama --version
    echo ""
    echo "To update Ollama, run: brew upgrade ollama"
    exit 0
fi

# Check if Homebrew is installed
if command -v brew &> /dev/null; then
    echo "📦 Homebrew detected. Installing Ollama via Homebrew..."
    brew install ollama
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Ollama installed successfully!"
        echo ""
        echo "To start Ollama, run:"
        echo "  ollama serve"
        echo ""
        echo "Or run it in the background:"
        echo "  brew services start ollama"
        echo ""
        echo "To pull a model, run:"
        echo "  ollama pull llama2"
        exit 0
    else
        echo "❌ Installation via Homebrew failed."
        exit 1
    fi
else
    echo "⚠️  Homebrew not found. Installing via official installer..."
    echo ""
    
    # Download and run the official Ollama installer
    curl -fsSL https://ollama.com/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Ollama installed successfully!"
        echo ""
        echo "To start Ollama, run:"
        echo "  ollama serve"
        echo ""
        echo "To pull a model, run:"
        echo "  ollama pull llama2"
        exit 0
    else
        echo "❌ Installation failed."
        echo ""
        echo "You can also install Ollama manually:"
        echo "1. Visit https://ollama.com/download"
        echo "2. Download the macOS installer"
        echo "3. Run the installer"
        exit 1
    fi
fi

