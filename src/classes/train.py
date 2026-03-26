import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from classes.income_classifier import IncomeClassifier


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s

    def forward(self, logits, targets):
        m = self.m_list.to(logits.device)
        margin = targets * m[1] + (1 - targets) * m[0]
        logits_adjusted = logits - margin
        return F.binary_cross_entropy_with_logits(self.s * logits_adjusted, targets)


def get_criterion(loss_type, y, device):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if loss_type == "weighted_bce":
        pos_weight = torch.tensor([np.sqrt(n_neg / n_pos)]).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        return FocalLoss(alpha=0.6, gamma=2.0)
    elif loss_type == "ldam":
        return LDAMLoss(cls_num_list=[n_neg, n_pos], max_m=0.5, s=30.0)
    else:
        return nn.BCEWithLogitsLoss()


def train_xgboost(X, y, args):
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.flatten()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(
        f"Class distribution: neg={n_neg} ({n_neg/len(y):.2%}), pos={n_pos} ({n_pos/len(y):.2%})"
    )

    model = XGBClassifier(
        scale_pos_weight=np.sqrt(n_neg / n_pos),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y, verbose=True)

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"{args.model_name}.pkl")
    joblib.dump(
        {
            "model": model,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "columns": args.columns,
            "model_type": "xgboost",
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


def train_mlp(X, y, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    input_dim = X.shape[1]

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(
        f"Class distribution: neg={n_neg} ({n_neg/len(y):.2%}), pos={n_pos} ({n_pos/len(y):.2%})"
    )

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    hidden_dim = 256
    model = IncomeClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    loss_type = getattr(args, "loss_type", "weighted_bce")
    criterion = get_criterion(loss_type, y, device)
    print(f"Loss function: {loss_type} ({criterion.__class__.__name__})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = (logits > 0).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / len(dataset)
        accuracy = correct / total * 100
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{args.epochs}, loss={avg_loss:.4f}, acc={accuracy:.2f}%, lr={scheduler.get_last_lr()[0]:.6f}"
        )

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"{args.model_name}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "scaler_mean": torch.tensor(scaler.mean_),
            "scaler_scale": torch.tensor(scaler.scale_),
            "columns": args.columns,
            "model_type": "mlp",
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


def train(X, y, args):
    model_type = getattr(args, "model_type", "mlp")
    if model_type == "xgboost":
        train_xgboost(X, y, args)
    else:
        train_mlp(X, y, args)
