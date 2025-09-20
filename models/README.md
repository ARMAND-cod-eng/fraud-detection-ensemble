# Models Directory

Trained model artifacts for the fraud detection ensemble.

## Model Files

1. **autoencoder.pth** (PyTorch)
   - Architecture: 30 → 32 → 24 → 14 → 24 → 32 → 30
   - Training: 50 epochs on normal transactions only
   - Purpose: Anomaly detection via reconstruction error

2. **isolation_forest.pkl** (Scikit-learn)
   - Trees: 200
   - Contamination: 0.002
   - Purpose: Statistical outlier detection

3. **xgboost_model.json** (XGBoost)
   - Trees: 300
   - Max depth: 6
   - Purpose: Final ensemble decision

## Model Performance
- Ensemble F1-Score: 0.6109
- Individual contributions shown in main README
