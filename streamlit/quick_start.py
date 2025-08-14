#!/usr/bin/env python3
"""
Quick Start Script for London City Classification Project

Automates the entire pipeline: data collection, training, evaluation, and web app.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_requirements():
    """Check if required files exist."""
    print("🔍 Checking project requirements...")
    
    required_files = [
        "google_maps_API.env",
        "data_collection/test_data_collector.py",
        "scripts/train_model.py",
        "scripts/evaluate_model.py",
        "streamlit_app.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    return True

def main():
    print("🚀 London City Classification - Quick Start")
    print("This script will automate the entire pipeline:")
    print("1. Collect test data")
    print("2. Train the model")
    print("3. Evaluate the model")
    print("4. Launch Streamlit web app")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix missing files before continuing.")
        return
    
    # Step 1: Collect test data
    print("\n📊 Step 1: Collecting test data...")
    if not run_command(
        "cd data_collection && python3 test_data_collector.py --mode quick --city both --winter 20 --outside 20",
        "Collecting test data"
    ):
        print("❌ Failed to collect test data. Continuing anyway...")
    
    # Step 2: Train model
    print("\n🤖 Step 2: Training model...")
    if not run_command(
        "cd scripts && python3 train_model.py",
        "Training model"
    ):
        print("❌ Failed to train model. Cannot continue.")
        return
    
    # Step 3: Evaluate model
    print("\n📈 Step 3: Evaluating model...")
    if not run_command(
        "cd scripts && python3 evaluate_model.py",
        "Evaluating model"
    ):
        print("⚠️  Failed to evaluate model. Continuing to web app...")
    
    # Step 4: Launch Streamlit app
    print("\n🌐 Step 4: Launching Streamlit web app...")
    print("The web app will open in your browser.")
    print("Press Ctrl+C to stop the app.")
    
    try:
        subprocess.run("streamlit run streamlit_app.py", shell=True)
    except KeyboardInterrupt:
        print("\n👋 Web app stopped. Thanks for using the London City Classifier!")

if __name__ == "__main__":
    main() 