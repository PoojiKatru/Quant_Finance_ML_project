"""
models.py — ML model definitions: XGBoost, LSTM, Random Forest, Ensemble.
All models follow a consistent sklearn-style interface.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ─────────────────────────────────────────────
# Base Model Wrapper
# ─────────────────────────────────────────────

class BaseModel:
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        proba = self.predict_proba(X_test)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
            "report": classification_report(y_test, preds, output_dict=True)
        }


# ─────────────────────────────────────────────
# XGBoost Model
# ─────────────────────────────────────────────

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = 0.05, subsample: float = 0.8):
        super().__init__("XGBoost")
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**self.params)
        except ImportError:
            # Fallback to sklearn GradientBoosting
            print("[XGBoost] xgboost not installed, using sklearn GradientBoosting as fallback.")
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self, feature_names: list) -> pd.Series:
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(
                self.model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
        return pd.Series()


# ─────────────────────────────────────────────
# Random Forest Model
# ─────────────────────────────────────────────

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 500, max_depth: int = 8,
                 min_samples_leaf: int = 20):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self, feature_names: list) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────────

class LSTMModel(BaseModel):
    def __init__(self, lookback: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 epochs: int = 50, batch_size: int = 64, lr: float = 1e-3):
        super().__init__("LSTM")
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model = None

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        sequences = []
        for i in range(self.lookback, len(X)):
            sequences.append(X[i - self.lookback:i])
        return np.array(sequences)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            X_scaled = self.scaler.fit_transform(X_train)
            X_seq = self._build_sequences(X_scaled)
            y_seq = y_train[self.lookback:]

            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y_seq)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            input_size = X_seq.shape[2]

            class LSTMClassifier(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        batch_first=True, dropout=dropout if num_layers > 1 else 0
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.attention = nn.Linear(hidden_size, 1)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 32),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                    context = (attn_weights * lstm_out).sum(dim=1)
                    return self.fc(context).squeeze()

            self._model = LSTMClassifier(input_size, self.hidden_size, self.num_layers, self.dropout)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            criterion = nn.BCELoss()

            self._model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    preds = self._model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                if (epoch + 1) % 10 == 0:
                    print(f"  [LSTM] Epoch {epoch+1}/{self.epochs} — Loss: {total_loss/len(loader):.4f}")

            self._torch = torch
            self.is_fitted = True

        except ImportError:
            print("[LSTM] PyTorch not available. Using Random Forest as fallback.")
            self._fallback = RandomForestModel()
            self._fallback.fit(X_train, y_train)
            self.is_fitted = True

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, "_fallback"):
            return self._fallback.predict_proba(X)

        import torch
        X_scaled = self.scaler.transform(X)
        X_seq = self._build_sequences(X_scaled)
        X_tensor = self._torch.FloatTensor(X_seq)
        self._model.eval()
        with self._torch.no_grad():
            proba = self._model(X_tensor).numpy()
        # Pad beginning with 0.5 for missing lookback
        pad = np.full(self.lookback, 0.5)
        return np.concatenate([pad, proba])


# ─────────────────────────────────────────────
# Ensemble Model
# ─────────────────────────────────────────────

class EnsembleModel(BaseModel):
    """
    Stacked ensemble: XGBoost + Random Forest + LSTM (if available).
    Final layer uses logistic regression meta-learner.
    """

    def __init__(self):
        super().__init__("Ensemble")
        self.base_models = {
            "xgboost": XGBoostModel(),
            "random_forest": RandomForestModel(),
        }
        self.meta_weights = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        print("Training base models...")

        # Use validation set for meta-learner, or split training set
        if X_val is None:
            split = int(0.8 * len(X_train))
            X_val, y_val = X_train[split:], y_train[split:]
            X_train, y_train = X_train[:split], y_train[:split]

        for name, model in self.base_models.items():
            print(f"  Fitting {name}...")
            model.fit(X_train, y_train)

        # Generate meta-features on val set
        meta_X = np.column_stack([
            m.predict_proba(X_val) for m in self.base_models.values()
        ])

        # Learn optimal blend weights via logistic regression
        from sklearn.linear_model import LogisticRegression
        self.meta_learner = LogisticRegression(C=1.0, random_state=42)
        self.meta_learner.fit(meta_X, y_val)
        self.meta_scaler = StandardScaler()
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_X = np.column_stack([
            m.predict_proba(X) for m in self.base_models.values()
        ])
        return self.meta_learner.predict_proba(meta_X)[:, 1]


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def get_model(model_type: str) -> BaseModel:
    models = {
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "lstm": LSTMModel,
        "ensemble": EnsembleModel,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    return models[model_type]()
