#  Notebooks Directory

This folder contains all Jupyter notebooks for the fraud detection analysis.

## Main Notebooks

###  01_fraud_detection_complete.ipynb
- **Purpose**: Complete end-to-end fraud detection analysis
- **Contents**: EDA, Feature Engineering, Model Training, Evaluation
- **Results**: 0.9834 ROC-AUC, 85.71% Recall, 1,779% ROI
- **Runtime**: ~10 minutes on Kaggle

### 📈 02_business_impact.ipynb  
- **Purpose**: Detailed ROI and business metrics analysis
- **Contents**: Threshold optimization, Cost-benefit analysis
- **Key Finding**: Optimal threshold of 0.5 yields 1,779% ROI

## How to Use

1. **On Kaggle** (Recommended): 
   - Visit https://www.kaggle.com/armandjunior
   - Click "Copy & Edit" to run with GPU

2. **Locally**:
```bash
   pip install -r requirements.txt
   jupyter notebook notebooks/01_fraud_detection_complete.ipynb
