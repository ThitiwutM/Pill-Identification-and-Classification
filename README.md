# Pill-Identification-and-Classification

# Pill Identification and Classification Using Hybrid Deep Learning Framework

## Overview

This project presents a Hybrid AI Framework for pill identification and classification using:

- YOLOv8s for object detection
- Random Forest for feature-based classification
- Feature Fusion using:
  - Color Features
  - Texture Features (LBP)
  - Shape Features
- Confidence-based Fusion Logic

The framework is designed to improve robustness and reduce false positive predictions in real-world environments.

---

## Features

✅ Pill Detection using YOLOv8s  
✅ Feature Extraction (Color, Texture, Shape)  
✅ Random Forest Classification  
✅ Confidence-based Hybrid Decision Fusion  
✅ Real-world Image Support  
✅ Streamlit Web Application

---

## System Architecture

Input Image  
↓  
YOLOv8s Detection  
↓  
ROI Cropping  
↓  
Feature Extraction  
(Color + Texture + Shape)  
↓  
Random Forest Classification  
↓  
Confidence-based Fusion Logic  
↓  
Final Prediction

---

## Dataset

The dataset contains:
- 10 medicine classes
- YOLO annotation format
- Train / Validation / Test split

---

## Models

### YOLOv8s
Used for:
- Object Detection
- ROI Localization

### Random Forest
Used for:
- Feature-based Classification Refinement

---

## Feature Extraction

### Color Features
- HSV Histogram

### Texture Features
- Local Binary Pattern (LBP)

### Shape Features
- Edge Density
- Aspect Ratio

---

## Fusion Logic

### Case 1
If YOLO confidence ≥ 0.95  
→ Trust YOLO prediction

### Case 2
If:
- YOLO confidence ≥ 0.60
- YOLO prediction == RF prediction
- RF confidence ≥ 0.60

→ Accept prediction

### Otherwise
→ Reject prediction

---

## Installation

```bash
git clone YOUR_GITHUB_REPOSITORY

cd pill-classification-hybrid
