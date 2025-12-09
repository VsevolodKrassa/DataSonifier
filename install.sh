#!/bin/bash
echo "Installing DataSonifier..."
echo

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.7 or later:"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "   macOS: brew install python"
    echo "   Windows: Download from python.org"
    echo "   or download from python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.7"

if [ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
    echo "Python $PYTHON_VERSION - OK"
else
    echo "ERROR: Python $PYTHON_VERSION < $REQUIRED_VERSION (requires 3.7+)"
    exit 1
fi

echo
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo
    echo "Installation completed successfully!"
    echo
    echo "Checking environment..."
    python3 check_environment.py
    
    echo
    echo "To run the program:"
    echo "   python3 datasonifier.py path/to/file.txt"
    echo
    echo "Or use the converter for non-PowerGraph data:"
    echo "   python3 converter.py your_data.csv"
else
    echo
    echo "ERROR: Failed to install dependencies"
    echo "Try: pip3 install --upgrade pip"
    echo "Or: python3 -m pip install -r requirements.txt"
    exit 1
fi
