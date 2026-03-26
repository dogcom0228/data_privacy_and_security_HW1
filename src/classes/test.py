import torch
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from classes.income_classifier import IncomeClassifier


def test_xgboost(X, y, args, ckpt):
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    X = scaler.transform(X).astype(np.float32)
    y = y.flatten()

    model = ckpt["model"]
    dmat = xgb.DMatrix(X)
    probs = model.get_booster().predict(dmat)
    preds = (probs > 0.5).astype(float)

    return y, preds, probs


def test_mlp(X, y, args, ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"].cpu().numpy()
    scaler.scale_ = ckpt["scaler_scale"].cpu().numpy()
    X = scaler.transform(X).astype(np.float32)

    model = IncomeClassifier(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(yb)

    all_probs = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    all_preds = (all_probs > 0.5).astype(float)

    return all_labels, all_preds, all_probs


def test(X, y, args):
    if args.model_path.endswith(".pkl"):
        ckpt = joblib.load(args.model_path)
        all_labels, all_preds, all_probs = test_xgboost(X, y, args, ckpt)
    else:
        ckpt = torch.load(args.model_path, map_location="cpu")
        all_labels, all_preds, all_probs = test_mlp(X, y, args, ckpt)

    acc = accuracy_score(all_labels, all_preds)
    misclass = 1 - acc
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*40}")
    print(f"Model: {args.model_path}")
    print(f"{'='*40}")
    print(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
    print(f"Misclassification:   {misclass:.4f} ({misclass*100:.2f}%)")
    print(f"Precision:           {prec:.4f}")
    print(f"Recall:              {rec:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"AUC:                 {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:>6d}  FP={cm[0][1]:>6d}")
    print(f"  FN={cm[1][0]:>6d}  TP={cm[1][1]:>6d}")
    print(f"{'='*40}")

    return {
        "accuracy": acc,
        "misclassification_rate": misclass,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }
