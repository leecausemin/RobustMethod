#!/bin/bash

echo "========================================================================"
echo "CNNDetection Dataset Download and Corruption Script"
echo "========================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        echo "Please install Python 3.7 or higher"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
echo ""

# Check if the script exists
if [ ! -f "download_cnndetection_and_corrupt.py" ]; then
    echo "Error: download_cnndetection_and_corrupt.py not found"
    echo "Please make sure you are running this script from the project root directory"
    exit 1
fi

# Run the Python script
echo "Starting CNNDetection dataset download and corruption process..."
echo ""

$PYTHON_CMD download_cnndetection_and_corrupt.py

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Script completed successfully!"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "Script failed with error code: $?"
    echo "========================================================================"
    exit 1
fi
