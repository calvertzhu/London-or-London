#!/bin/bash

echo "Setting up London City Classification Project"
echo "============================================="

# Check if virtual environment exists
if [ ! -d "aps360env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv aps360env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source aps360env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if Google Maps API key file exists
if [ ! -f "google_maps_API.env" ]; then
    echo "Warning: google_maps_API.env not found!"
    echo "Please create this file with your Google Maps API key:"
    echo "GOOGLE_MAPS_API_KEY=your_api_key_here"
fi

echo ""
echo "Setup complete!"
echo "To activate the environment, run: source aps360env/bin/activate"
echo "To run the project, make sure you have:"
echo "1. Google Maps API key in google_maps_API.env"
echo "2. Collected training data in data/ directory"
echo "3. Collected test data using test_data_collector.py" 