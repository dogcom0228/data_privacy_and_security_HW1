# Data Privacy and Security

Investigating the **privacy-utility tradeoff** of Mondrian k-anonymity on the UCI Adult (Census Income) dataset. We anonymize the dataset with varying k values, then train and evaluate ML models to measure how anonymization affects classification performance.

## Project Structure

```
├── src/
│   ├── main.py                    # CLI entry point (mondrian / train / test)
│   ├── classes/
│   │   ├── income_classifier.py   # MLP model with residual connections
│   │   ├── train.py               # Training logic (MLP & XGBoost)
│   │   └── test.py                # Evaluation logic
│   ├── algorithms/
│   │   └── mondrian.py            # Mondrian k-anonymity implementation
│   ├── utils/
│   │   └── tools.py               # Data loading & preprocessing
│   ├── scripts/
│   │   ├── train.sh               # Batch training (MLP)
│   │   ├── test.sh                # Batch testing (MLP)
│   │   ├── train_xgboost.sh       # Batch training (XGBoost)
│   │   └── test_xgboost.sh        # Batch testing (XGBoost)
│   ├── data/
│   │   ├── adult.data             # Training set
│   │   └── adult.test             # Test set
│   ├── models/                    # Saved model checkpoints
│   └── results/                   # Anonymized datasets
├── environment.yml
└── requirements.txt
```

## Setup

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate dps-hw1
```

### Option 2: pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

All commands are run from the `src/` directory.

### 1. Anonymize Data

Apply Mondrian k-anonymity with a given k value:

```bash
python main.py mondrian --input_file data/adult.data --k 100
```

Output is saved to `results/anonymized_data_k100.csv`.

### 2. Train Models

**MLP:**

```bash
python main.py train \
    --dataset data/adult.data \
    --model_name adult \
    --model_type mlp \
    --loss_type weighted_bce \
    --epochs 50 --batch_size 512 --lr 0.0003
```

**XGBoost:**

```bash
python main.py train \
    --dataset data/adult.data \
    --model_name adult_xgb \
    --model_type xgboost
```

**Batch training** (all k values):

```bash
bash scripts/train.sh            # MLP
bash scripts/train_xgboost.sh    # XGBoost
```

### 3. Evaluate Models

```bash
python main.py test \
    --model_path models/adult.pt \
    --test_data data/adult.test \
    --batch_size 256
```

**Batch testing:**

```bash
bash scripts/test.sh             # MLP
bash scripts/test_xgboost.sh     # XGBoost
```

## Models

### MLP (IncomeClassifier)

- 2-block residual MLP with hidden dimension 256
- BatchNorm + ReLU + Dropout (0.3)
- Weighted BCE loss with `pos_weight = sqrt(n_neg / n_pos)`
- CosineAnnealing LR scheduler

### XGBoost

- 300 estimators, max depth 6
- `scale_pos_weight = sqrt(n_neg / n_pos)`
- GPU-accelerated (`tree_method="hist"`, `device="cuda"`)

## Evaluation Metrics

| Metric                 | Description                                         |
|------------------------|-----------------------------------------------------|
| Accuracy               | Overall correct predictions / total samples         |
| Misclassification Rate | 1 - Accuracy                                        |
| Precision              | TP / (TP + FP)                                      |
| Recall                 | TP / (TP + FN)                                      |
| F1 Score               | Harmonic mean of Precision and Recall               |
| AUC                    | Area under the ROC curve (probability-based)        |

## Results

Performance on `adult.test` across different k-anonymity levels:

### MLP

| k         | Accuracy | Misclass | Precision | Recall | F1     |
|-----------|----------|----------|-----------|--------|--------|
| Baseline  | 84.25%   | 15.75%   | 64.32%    | 74.80% | 69.17% |
| 10        | 81.15%   | 18.85%   | 57.67%    | 75.95% | 65.56% |
| 25        | 82.79%   | 17.21%   | 62.78%    | 66.67% | 64.67% |
| 50        | 82.32%   | 17.68%   | 62.55%    | 62.74% | 62.64% |
| 75        | 81.46%   | 18.54%   | 59.67%    | 66.41% | 62.86% |
| 100       | 82.44%   | 17.56%   | 61.05%    | 70.90% | 65.61% |
| 1,000     | 78.50%   | 21.50%   | 54.18%    | 58.35% | 56.18% |
| 10,000    | 76.50%   | 23.50%   | 50.20%    | 64.07% | 56.29% |

### XGBoost

| k         | Accuracy | Misclass | Precision | Recall | F1     |
|-----------|----------|----------|-----------|--------|--------|
| Baseline  | 86.21%   | 13.79%   | 68.77%    | 76.26% | 72.32% |
| 10        | 85.96%   | 14.04%   | 69.14%    | 73.27% | 71.14% |
| 25        | 85.39%   | 14.61%   | 67.57%    | 73.40% | 70.36% |
| 50        | 85.71%   | 14.29%   | 71.16%    | 66.46% | 68.73% |
| 75        | 85.33%   | 14.67%   | 72.82%    | 60.45% | 66.06% |
| 100       | 85.00%   | 15.00%   | 86.30%    | 43.40% | 57.75% |
| 1,000     | 83.58%   | 16.42%   | 72.97%    | 48.44% | 58.23% |
| 10,000    | 83.44%   | 16.56%   | 71.44%    | 49.82% | 58.70% |

## Dataset

[UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) — 48,842 samples, 14 features, binary classification (income >50K or ≤50K).

## License

This project is for academic use only.
