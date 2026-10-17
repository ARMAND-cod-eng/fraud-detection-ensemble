# Data

Datasets are **not committed** (size and licence). `data/raw/` is git-ignored.

## ULB credit-card fraud (primary benchmark)

* Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud — Machine Learning Group, ULB and
  Worldline; Database Contents License.
* 284,807 transactions by European cardholders over two days in September 2013; 492 frauds
  (0.172%); features V1–V28 (PCA, meaning not disclosed), `Amount`, `Time` (seconds since the first
  transaction), label `Class`.
* 1,081 exact duplicate rows (19 fraud) — removed by the pipeline before splitting.

```bash
fraudshield download-data
```

uses the Kaggle CLI if you have configured your own API token, otherwise the public copy used by
the TensorFlow imbalanced-data tutorial; the file is accepted only if its SHA-256 equals
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

## IEEE-CIS fraud detection (second benchmark, optional)

Requires accepting the competition rules on Kaggle with your own account:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-cis
unzip data/raw/ieee-cis/ieee-fraud-detection.zip -d data/raw/ieee-cis
python benchmarks/ieee_cis_benchmark.py --root data/raw/ieee-cis
```

Its results are reported separately and are never combined with the credit-card benchmark.
