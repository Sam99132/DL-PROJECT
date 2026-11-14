"""
feedforward_nn.py - Normal Neural Network (MLP & Residual MLP) for Inventory Demand Forecasting.

Implements standard Feedforward Deep Neural Networks with Entity Embeddings,
Batch Normalization, Dropout Regularization, and Non-Negative Softplus Output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class NormalDemandMLP(nn.Module):
    """
    Standard Feedforward Neural Network (Multi-Layer Perceptron) with Entity Embeddings
    for Tabular Retail Time-Series Demand Forecasting.

    Mathematical Formulation:
      1. e_k = Embedding_k(c_k) for categorical entity k
      2. h_0 = [e_1 || ... || e_K || x_cont]
      3. For l in 1...L:
           z_l = W_l * h_{l-1} + b_l
           a_l = BatchNorm1d(z_l)  (optional)
           h_l = Dropout(Activation(a_l), p)
      4. y_hat = Softplus(W_out * h_L + b_out) >= 0
    """

    def __init__(
        self,
        cat_dims: Dict[str, int],
        num_cont_features: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        embedding_dim_strategy: str = "rule_of_thumb",
        custom_emb_dims: Optional[Dict[str, int]] = None,
        dropout_rate: float = 0.20,
        use_batch_norm: bool = True,
        activation: str = "gelu",
        use_softplus_output: bool = True,
    ):
        super().__init__()
        self.cat_dims = cat_dims
        self.num_cont_features = num_cont_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_softplus_output = use_softplus_output

        # 1. Construct Learnable Entity Embeddings
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        for col_name, num_classes in cat_dims.items():
            if custom_emb_dims and col_name in custom_emb_dims:
                emb_dim = custom_emb_dims[col_name]
            elif embedding_dim_strategy == "rule_of_thumb":
                # Standard empirical heuristic: min(50, ceil(num_classes^0.5 * 2))
                emb_dim = min(50, max(4, int(math.ceil(math.sqrt(num_classes) * 2))))
            else:
                emb_dim = min(32, max(4, num_classes // 2))

            self.embeddings[col_name] = nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=emb_dim,
            )
            # Initialize weights with standard normal Xavier/Kaiming scale
            nn.init.kaiming_normal_(self.embeddings[col_name].weight, mode="fan_in", nonlinearity="linear")
            total_emb_dim += emb_dim

        # Input dimension is sum of embedding dimensions plus continuous features
        self.in_dim = total_emb_dim + num_cont_features

        # 2. Activation function selector
        if activation.lower() == "gelu":
            act_fn = nn.GELU()
        elif activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 3. Hidden Layers Stack
        layers = []
        prev_dim = self.in_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(act_fn)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = dim

        self.hidden_network = nn.Sequential(*layers)

        # 4. Output Projection Layer (Predicting scalar continuous demand)
        self.head = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
          cat_features: (Batch_Size, Num_Categorical_Features) LongTensor
          cont_features: (Batch_Size, Num_Continuous_Features) FloatTensor
        Returns:
          y_pred: (Batch_Size,) Non-negative predicted sales demand
        """
        # Lookup and concatenate entity embeddings
        emb_list = []
        for i, (col_name, emb_layer) in enumerate(self.embeddings.items()):
            col_idx = cat_features[:, i]
            emb = emb_layer(col_idx)
            emb_list.append(emb)

        if emb_list:
            x_cat = torch.cat(emb_list, dim=-1)
            x = torch.cat([x_cat, cont_features], dim=-1)
        else:
            x = cont_features

        # Forward through dense layers
        h = self.hidden_network(x)
        raw_out = self.head(h).squeeze(-1)

        # Output activation: strictly non-negative demand
        if self.use_softplus_output:
            return F.softplus(raw_out)
        else:
            return F.relu(raw_out)


class ResidualDemandBlock(nn.Module):
    """
    Residual dense block with identity skip connection:
    y = x + Dropout(Linear(Act(BatchNorm(Linear(x)))))
    """

    def __init__(self, dim: int, dropout_rate: float = 0.2, act_fn: nn.Module = nn.GELU()):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = act_fn
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        return self.act(residual + out)


class ResidualDemandMLP(nn.Module):
    """
    Deep Residual Multi-Layer Perceptron for complex retail demand interactions.
    Applies residual skip connections to facilitate deep representation learning.
    """

    def __init__(
        self,
        cat_dims: Dict[str, int],
        num_cont_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.20,
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        for col_name, num_classes in cat_dims.items():
            emb_dim = min(50, max(4, int(math.ceil(math.sqrt(num_classes) * 2))))
            self.embeddings[col_name] = nn.Embedding(num_classes, emb_dim)
            total_emb_dim += emb_dim

        self.in_dim = total_emb_dim + num_cont_features

        # Projection into residual dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Residual stack
        self.blocks = nn.ModuleList([
            ResidualDemandBlock(dim=hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        # Head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> torch.Tensor:
        emb_list = [emb_layer(cat_features[:, i]) for i, (k, emb_layer) in enumerate(self.embeddings.items())]
        x = torch.cat(emb_list + [cont_features], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        raw_out = self.head(h).squeeze(-1)
        return F.softplus(raw_out)


class TweedieDevianceLoss(nn.Module):
    """
    Tweedie Deviance Loss for Compound Poisson-Gamma Intermittent Demand.
    Parameter p in (1, 2) balances zero-inflation mass (Poisson) and positive volume (Gamma).
    Default p=1.5 is standard for retail demand (M5 competition winner standard).
    """

    def __init__(self, p: float = 1.5, eps: float = 1e-6):
        super().__init__()
        assert 1.0 < p < 2.0, "Tweedie variance power p must be in (1, 2)"
        self.p = p
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, min=self.eps)
        p = self.p
        a = y_true * torch.pow(y_pred, 1.0 - p) / (1.0 - p)
        b = torch.pow(y_pred, 2.0 - p) / (2.0 - p)
        loss = 2.0 * (b - a)
        return torch.mean(loss)
