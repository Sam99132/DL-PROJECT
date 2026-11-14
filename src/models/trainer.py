"""
trainer.py - Rigorous Deep Learning Trainer, Optimizer & Evaluator for Demand Forecasting.

Handles training loops, learning rate scheduling, early stopping, gradient clipping,
and evaluation metrics: RMSE, MAE, WAPE, RMSSE, and R2.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_historical: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes rigorous retail forecasting metrics:
      - RMSE (Root Mean Squared Error)
      - MAE (Mean Absolute Error)
      - WAPE (Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y))
      - RMSSE (Root Mean Squared Scaled Error - official M5 competition metric)
      - R2 (Coefficient of Determination)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 1. RMSE
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # 2. MAE
    mae = float(np.mean(np.abs(y_pred - y_true)))

    # 3. WAPE (Avoids division by zero on individual zero demand days)
    sum_true = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_pred - y_true)) / (sum_true + 1e-8)) * 100.0

    # 4. R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    # 5. RMSSE (M5 metric: scales MSE against historical 1-lag random walk MSE)
    if y_train_historical is not None and len(y_train_historical) > 1:
        scale_denom = np.mean(np.diff(y_train_historical) ** 2)
        rmsse = float(np.sqrt(np.mean((y_true - y_pred) ** 2) / (scale_denom + 1e-8)))
    else:
        # Fallback default scaling
        rmsse = rmse / (np.std(y_true) + 1e-8)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "wape_pct": round(wape, 2),
        "rmsse": round(rmsse, 4),
        "r2": round(r2, 4),
    }


class DemandTrainer:
    """
    Orchestrates the training, validation, checkpointing, and metric tracking
    for the Normal Neural Network.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.HuberLoss(delta=1.0),
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        clip_grad_norm: float = 1.0,
        checkpoint_dir: str = "DL-PROJECT/reports/checkpoints",
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # AdamW with Decoupled Weight Decay Regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_wape": [],
            "learning_rate": [],
        }
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict[str, Any]] = None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Trains model for one full epoch over the dataloader."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for cat_feat, cont_feat, targets in dataloader:
            cat_feat = cat_feat.to(self.device)
            cont_feat = cont_feat.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(cat_feat, cont_feat)
            loss = self.criterion(preds, targets)

            loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluates model performance and computes standard metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for cat_feat, cont_feat, targets in dataloader:
                cat_feat = cat_feat.to(self.device)
                cont_feat = cont_feat.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(cat_feat, cont_feat)
                loss = self.criterion(preds, targets)

                total_loss += loss.item()
                n_batches += 1

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(1, n_batches)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        metrics = calculate_metrics(y_true, y_pred)

        return avg_loss, metrics, y_true, y_pred

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        early_stopping_patience: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Executes full training with early stopping and performance logging."""
        start_time = time.time()
        patience_counter = 0

        if verbose:
            print(f"[TRAINER] Commencing training on device '{self.device}' for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics, _, _ = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse"].append(metrics["rmse"])
            self.history["val_mae"].append(metrics["mae"])
            self.history["val_wape"].append(metrics["wape_pct"])
            self.history["learning_rate"].append(current_lr)

            elapsed = time.time() - t0

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | "
                    f"WAPE: {metrics['wape_pct']:.2f}% | LR: {current_lr:.1e}"
                )

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_state, self.checkpoint_dir / "best_normal_demand_mlp.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"[EARLY STOPPING] Validation loss did not improve for {early_stopping_patience} epochs. Stopping.")
                    break

        # Load best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        total_time = time.time() - start_time
        final_val_loss, final_metrics, y_true, y_pred = self.evaluate(val_loader)

        return {
            "total_time_seconds": round(total_time, 2),
            "epochs_trained": len(self.history["train_loss"]),
            "best_val_loss": round(self.best_val_loss, 4),
            "final_metrics": final_metrics,
            "history": self.history,
            "y_true": y_true,
            "y_pred": y_pred,
        }
