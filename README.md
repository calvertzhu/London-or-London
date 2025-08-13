APS360 Project

# 🌍 London vs. London: Street View Image Classifier

A deep learning project that distinguishes between Google Street View images from **London, UK** and **London, Ontario (Canada)** using high-resolution panoramic imagery and a custom-trained convolutional neural network.

---

## 🧠 Project Overview

Two cities. One name. How to tell them apart?

This project builds and trains a binary image classifier to distinguish between images from two cities that share the same name but have vastly different visual characteristics. The model is trained on panoramic Street View data collected via the **Google Maps Static Street View API**, then evaluated for accuracy and generalization across both urban environments.

---

## 🔧 Features

- ✅ Collects real-world panoramas using GPS sampling and the Google Street View API
- ✅ Preprocesses and compresses images to 224×224 JPEGs
- Developing: Filters near-duplicate images with spatial indexing and perceptual hashes 
- Developing: Trains a ResNet-50 classifier using PyTorch and transfer learning 
- Developing: Visualizes predictions using Grad-CAM heatmaps 
- Developing: Web front end via Streamlit for uploading and classifying new images 

---

## 📦 Installation

### Quick Setup
```bash
git clone https://github.com/calvertzhu/london-vs-london.git
cd london-vs-london
./setup.sh
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv aps360env
source aps360env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or for minimal installation:
pip install -r requirements-minimal.txt
```

### API Key Setup
Create a `google_maps_API.env` file in the project root:
```bash
echo "GOOGLE_MAPS_API_KEY=your_api_key_here" > google_maps_API.env
```

### Dependencies
- **Core ML**: PyTorch, TorchVision, NumPy, Pandas
- **Image Processing**: Pillow, OpenCV
- **Data Collection**: Google Street View API, Requests
- **Visualization**: Matplotlib, Seaborn

## 🚀 Usage

### Data Collection
```bash
# Collect test data
cd data_collection
python3 test_data_collector.py --mode comprehensive --city both --winter 50 --outside 50

# Analyze existing data
python3 analyze_test_needs.py
```

### Training
```bash
# Split data for training
python3 scripts/data_splitter.py

# Train the model
python3 scripts/train_model.py
```

### Project Structure
```
├── data_collection/     # Data collection scripts
├── data/               # Training data
├── test_data/          # Test data
├── models/             # Model architectures
├── scripts/            # Training and evaluation scripts
├── report_data/        # Split train/val data
└── metadata/           # Data metadata and analysis
```