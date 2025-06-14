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
- ✅ Filters near-duplicate images with spatial indexing and perceptual hashes (developing)
- ✅ Trains a ResNet-50 classifier using PyTorch and transfer learning (developing)
- ✅ Visualizes predictions using Grad-CAM heatmaps (developing)
- ✅ Web front end via Streamlit for uploading and classifying new images (developing)

---

## 📦 Installation

bash
git clone https://github.com/calvertzhu/london-vs-london.git
cd london-vs-london
pip install -r requirements.txt

You'll also need to export your Google Maps API key:
export GOOGLE_MAPS_API_KEY="your_api_key_here"