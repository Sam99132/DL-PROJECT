"""
ablation.py - Automated Scientific Ablation Study Framework for Phase 2 Evaluation.

Conducts controlled experimental comparisons across:
1. Embedding Representations (Learned Embeddings vs. Raw/One-Hot)
2. Regularization Strategies (Dropout + BatchNorm + Weight Decay vs. Ablated)
3. Architecture Depth/Width (Shallow vs. Standard Deep vs. Residual MLP)
4. Loss Formulations (Huber vs. MSE vs. Tweedie Deviance)
"""

import json
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List

from src.models.feedforward_nn import NormalDemandMLP, ResidualDemandMLP, TweedieDevianceLoss
from src.models.trainer import DemandTrainer


class M5AblationRunner:
    """Orchestrates and logs all 4 required Phase 2 ablation studies."""

    def __init__(
        self,
        cat_dims: Dict[str, int],
        num_cont_features: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs_per_trial: int = 5,
        output_dir: str = "DL-PROJECT/reports",
    ):
        self.cat_dims = cat_dims
        self.num_cont_features = num_cont_features
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs_per_trial = epochs_per_trial
        project_root = Path(__file__).resolve().parent.parent.parent
        out_p = Path(output_dir)
        if not out_p.is_absolute() and not out_p.exists():
            for cand in [Path("reports"), project_root / "reports", project_root / output_dir]:
                if cand.exists():
                    out_p = cand
                    break
        self.output_dir = out_p
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}

    def run_all_ablations(self) -> Dict[str, Any]:
        """Runs the complete suite of 4 ablation experiments."""
        print("\n" + "=" * 70)
        print("🚀 COMMENCING PHASE 2 DEEP LEARNING ABLATION SUITE")
        print("=" * 70)

        self.results["ablation_1_embeddings"] = self._run_embedding_ablation()
        self.results["ablation_2_regularization"] = self._run_regularization_ablation()
        self.results["ablation_3_architecture"] = self._run_architecture_ablation()
        self.results["ablation_4_loss_functions"] = self._run_loss_ablation()

        # Save to JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print("\n" + "=" * 70)
        print(f"✅ ABLATION EXPERIMENTS COMPLETED. Results written to {json_path}")
        print("=" * 70)
        return self.results

    def _run_embedding_ablation(self) -> List[Dict[str, Any]]:
        print("\n[ABLATION 1] Evaluating Entity Embeddings vs. Minimal Embeddings...")
        configs = [
            ("Standard Entity Embeddings", {"embedding_dim_strategy": "rule_of_thumb"}),
            ("Low-Dimensional Embeddings", {"embedding_dim_strategy": "minimal"}),
        ]
        trial_results = []
        for name, kwargs in configs:
            print(f"  --> Running configuration: '{name}'")
            model = NormalDemandMLP(
                cat_dims=self.cat_dims,
                num_cont_features=self.num_cont_features,
                hidden_dims=[256, 128, 64],
                dropout_rate=0.20,
                use_batch_norm=True,
                **kwargs,
            )
            trainer = DemandTrainer(model=model, criterion=nn.HuberLoss(delta=1.0), lr=1e-3)
            out = trainer.fit(self.train_loader, self.val_loader, epochs=self.epochs_per_trial, verbose=False)
            trial_results.append({
                "configuration": name,
                "val_rmse": out["final_metrics"]["rmse"],
                "val_mae": out["final_metrics"]["mae"],
                "val_wape_pct": out["final_metrics"]["wape_pct"],
                "val_rmsse": out["final_metrics"]["rmsse"],
            })
            print(f"      Result: RMSE={out['final_metrics']['rmse']:.4f}, WAPE={out['final_metrics']['wape_pct']:.2f}%")
        return trial_results

    def _run_regularization_ablation(self) -> List[Dict[str, Any]]:
        print("\n[ABLATION 2] Evaluating Regularization Mechanisms (Dropout & BatchNorm)...")
        configs = [
            ("Full Regularization (Dropout=0.2, BatchNorm=True, L2=1e-4)", {"dropout_rate": 0.20, "use_batch_norm": True, "weight_decay": 1e-4}),
            ("No Dropout (Dropout=0.0, BatchNorm=True, L2=1e-4)", {"dropout_rate": 0.0, "use_batch_norm": True, "weight_decay": 1e-4}),
            ("No BatchNorm (Dropout=0.2, BatchNorm=False, L2=1e-4)", {"dropout_rate": 0.20, "use_batch_norm": False, "weight_decay": 1e-4}),
            ("Unregularized Baseline (Dropout=0.0, BatchNorm=False, L2=0.0)", {"dropout_rate": 0.0, "use_batch_norm": False, "weight_decay": 0.0}),
        ]
        trial_results = []
        for name, cfg in configs:
            print(f"  --> Running configuration: '{name}'")
            model = NormalDemandMLP(
                cat_dims=self.cat_dims,
                num_cont_features=self.num_cont_features,
                hidden_dims=[256, 128, 64],
                dropout_rate=cfg["dropout_rate"],
                use_batch_norm=cfg["use_batch_norm"],
            )
            trainer = DemandTrainer(
                model=model,
                criterion=nn.HuberLoss(delta=1.0),
                lr=1e-3,
                weight_decay=cfg["weight_decay"],
            )
            out = trainer.fit(self.train_loader, self.val_loader, epochs=self.epochs_per_trial, verbose=False)
            trial_results.append({
                "configuration": name,
                "val_rmse": out["final_metrics"]["rmse"],
                "val_mae": out["final_metrics"]["mae"],
                "val_wape_pct": out["final_metrics"]["wape_pct"],
                "val_rmsse": out["final_metrics"]["rmsse"],
            })
            print(f"      Result: RMSE={out['final_metrics']['rmse']:.4f}, WAPE={out['final_metrics']['wape_pct']:.2f}%")
        return trial_results

    def _run_architecture_ablation(self) -> List[Dict[str, Any]]:
        print("\n[ABLATION 3] Evaluating Network Depth & Width Architectures...")
        trial_results = []

        # 1. Shallow MLP
        print("  --> Running: 'Shallow MLP (2 Layers: [128, 64])'")
        m_shallow = NormalDemandMLP(
            cat_dims=self.cat_dims,
            num_cont_features=self.num_cont_features,
            hidden_dims=[128, 64],
            dropout_rate=0.20,
            use_batch_norm=True,
        )
        t_shallow = DemandTrainer(m_shallow, nn.HuberLoss(), lr=1e-3)
        res_shallow = t_shallow.fit(self.train_loader, self.val_loader, epochs=self.epochs_per_trial, verbose=False)
        trial_results.append({
            "configuration": "Shallow MLP (2 Layers: [128, 64])",
            "val_rmse": res_shallow["final_metrics"]["rmse"],
            "val_mae": res_shallow["final_metrics"]["mae"],
            "val_wape_pct": res_shallow["final_metrics"]["wape_pct"],
            "val_rmsse": res_shallow["final_metrics"]["rmsse"],
        })
        print(f"      Result: RMSE={res_shallow['final_metrics']['rmse']:.4f}")

        # 2. Standard Deep MLP
        print("  --> Running: 'Standard Deep MLP (4 Layers: [256, 128, 64, 32])'")
        m_deep = NormalDemandMLP(
            cat_dims=self.cat_dims,
            num_cont_features=self.num_cont_features,
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=0.20,
            use_batch_norm=True,
        )
        t_deep = DemandTrainer(m_deep, nn.HuberLoss(), lr=1e-3)
        res_deep = t_deep.fit(self.train_loader, self.val_loader, epochs=self.epochs_per_trial, verbose=False)
        trial_results.append({
            "configuration": "Standard Deep MLP (4 Layers: [256, 128, 64, 32])",
            "val_rmse": res_deep["final_metrics"]["rmse"],
            "val_mae": res_deep["final_metrics"]["mae"],
            "val_wape_pct": res_deep["final_metrics"]["wape_pct"],
            "val_rmsse": res_deep["final_metrics"]["rmsse"],
        })
        print(f"      Result: RMSE={res_deep['final_metrics']['rmse']:.4f}")

        # 3. Deep Residual MLP
        print("  --> Running: 'Residual MLP (3 Blocks, Dim=128)'")
        m_res = ResidualDemandMLP(
            cat_dims=self.cat_dims,
            num_cont_features=self.num_cont_features,
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.20,
        )
        t_res = DemandTrainer(m_res, nn.HuberLoss(), lr=1e-3)
        res_res = t_res.fit(self.train_loader, self.val_loader, epochs=self.epochs_per_trial, verbose=False)
        trial_results.append({
            "configuration": "Residual MLP (3 Blocks, Dim=128)",
            "val_rmse": res_res["final_metrics"]["rmse"],
            "val_mae": res_res["final_metrics"]["mae"],
            "val_wape_pct": res_res["final_metrics"]["wape_pct"],
            "val_rmsse": res_res["final_metrics"]["rmsse"],
        })
        print(f"      Result: RMSE={res_res['final_metrics']['rmse']:.4f}")

        return trial_results

    def _run_loss_ablation(self) -> List[Dict[str, Any]]:
        print("\n[ABLATION 4] Evaluating Objective Loss Formulations...")
        losses = [
            ("Huber / Smooth L1 Loss (delta=1.0)", nn.HuberLoss(delta=1.0)),
            ("Mean Squared Error (MSE Loss)", nn.MSELoss()),
            ("Tweedie Deviance Loss (p=1.5)", TweedieDevianceLoss(p=1.5)),
        ]
        trial_results = []
        for name, loss_fn in losses:
            print(f"  --> Running loss formulation: '{name}'")
            model = NormalDemandMLP(
                cat_dims=self.cat_dims,
                num_cont_features=self.num_cont_features,
                hidden_dims=[256, 128, 64],
                dropout_rate=0.20,
                use_batch_norm=True,
            )
            trainer = DemandTrainer(model=model, criterion=loss_fn, lr=1e-3)
            out = trainer.fit(self.train_loader, self.val_loader, epochs=self.epochs_per_trial, verbose=False)
            trial_results.append({
                "configuration": name,
                "val_rmse": out["final_metrics"]["rmse"],
                "val_mae": out["final_metrics"]["mae"],
                "val_wape_pct": out["final_metrics"]["wape_pct"],
                "val_rmsse": out["final_metrics"]["rmsse"],
            })
            print(f"      Result: RMSE={out['final_metrics']['rmse']:.4f}, WAPE={out['final_metrics']['wape_pct']:.2f}%")
        return trial_results
