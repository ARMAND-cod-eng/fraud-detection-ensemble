# 🛡️ Real-Time Fraud Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/armandjunior/real-time-fraud-detection-system-using-autoencoder)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5)](https://www.linkedin.com/in/notue250/)
[![ROI](https://img.shields.io/badge/ROI-1,779%25-success)](https://github.com/ARMAND-cod-eng/fraud-detection-ensemble)
[![Recall](https://img.shields.io/badge/Recall-85.71%25-green)](https://github.com/ARMAND-cod-eng/fraud-detection-ensemble)
[![FPR](https://img.shields.io/badge/FPR-0.16%25-orange)](https://github.com/ARMAND-cod-eng/fraud-detection-ensemble)

<h3>🎯 Production-Ready Fraud Detection achieving 1,779% ROI</h3>

</div>

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Business Impact](#business-impact)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 🎯 Overview

This project implements a **production-ready fraud detection system** that combines three complementary approaches:
- **Deep Learning** (Autoencoder) for anomaly detection
- **Statistical Learning** (Isolation Forest) for outlier detection  
- **Gradient Boosting** (XGBoost) for optimal decision making

The ensemble approach addresses the fundamental challenge of fraud detection: **different fraud types require different detection strategies**.

---

## 🏆 Key Results

### Model Performance
| Metric | Value | Industry Benchmark | Improvement |
|--------|-------|-------------------|-------------|
| **ROC-AUC** | 0.9834 | 0.95-0.97 | +3.4% |
| **Recall** | 85.71% | 70-80% | +16% |
| **Precision** | 47.46% | 40-50% | Comparable |
| **F1-Score** | 0.6109 | 0.50-0.60 | +22% |
| **FPR** | 0.16% | 1-3% | **10x better** |

### Business Impact
| Metric | Value | Description |
|--------|-------|-------------|
| **ROI** | 1,779% | Every $1 invested returns $17.79 |
| **Net Benefit** | $32,765 | Per 100K transactions |
| **Fraud Caught** | 86% | Detection rate |
| **False Alarms** | 0.16% | Only 1 in 625 |
| **Processing** | 86ms | Real-time capable |

---

## 🚀 Quick Start

### Option 1: Run on Kaggle (Recommended)
[![Open in Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF)](https://www.kaggle.com/armandjunior/fraud-detection-ensemble)

1. Click the badge above
2. Click "Copy & Edit" 
3. Run all cells (GPU recommended)

### Option 2: Run Locally
```bash
# Clone repository
git clone https://github.com/ARMAND-cod-eng/fraud-detection-ensemble.git
cd fraud-detection-ensemble

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/01_fraud_detection_complete.ipynb
```


## 📁 Project Structure

fraud-detection-ensemble/
│
├── 📓 notebooks/
│   ├── 01_fraud_detection_complete.ipynb  # Main analysis (from Kaggle)
│   └── README.md                          # Notebook descriptions
│
├── 🤖 models/
│   └── README.md                          # Model documentation
│
├── 📊 data/
│   └── README.md                          # Data information
│
├── 📝 src/
│   └── README.md                          # Source code (coming soon)
│
├── 📋 requirements.txt                    # Python dependencies
├── 📄 LICENSE                             # MIT License
└── 📄 README.md                          # This file

## 🏗️ Model Architecture
### Ensemble Strategy

Input Transaction
       ↓
┌──────────────────────────────────────────┐
│          Feature Engineering              │
│        (44 features from 30 PCA)          │
└──────────────────────────────────────────┘
       ↓                ↓                ↓
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Autoencoder │ │  Isolation  │ │   Original   │
│  (PyTorch)  │ │   Forest    │ │   Features   │
└─────────────┘ └─────────────┘ └─────────────┘
       ↓                ↓                ↓
    AE Score      ISO Score         Features
       └────────────┬────────────┘
                    ↓
           ┌─────────────────┐
           │     XGBoost     │
           │  (Final Decision)│
           └─────────────────┘
                    ↓
            Fraud Probability

### Setup

# Create virtual environment
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # Windows: fraud_detection_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/


### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

- Fork the repository
- Create your feature branch (git checkout -b feature/AmazingFeature)
- Commit changes (git commit -m 'Add AmazingFeature')
- Push to branch (git push origin feature/AmazingFeature)
- Open a Pull Request

##  Contact
Armand Junior Dongmo Notue

📧 Email: notuearmand250@gmail.com
💼 LinkedIn: linkedin.com/in/notue250
🔗 GitHub: @ARMAND-cod-eng
📊 Kaggle: https://www.kaggle.com/armandjunior


### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments

Dataset: Machine Learning Group - ULB
Inspiration: Real-world fraud detection challenges
