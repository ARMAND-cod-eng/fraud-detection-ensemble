# Data Directory

## Dataset Information

**Source**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Context**
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

**Content**
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

**Statistics**:
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Features: 30 (28 PCA + Time + Amount)
- Timespan: 48 hours

## File Structure
- `raw/` - Original dataset (not included due to size)
- `processed/` - Preprocessed data files
- `external/` - External data sources

## Download Instructions
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/
