from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

try:
    from .models import TechnicalSignal
except ImportError:
    from models import TechnicalSignal


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_features: int,
        cnn_channels: Optional[List[int]] = None,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        num_classes: int = 2,
        use_regression_head: bool = False,
    ):
        super().__init__()
        self.use_regression_head = use_regression_head

        cnn_channels = cnn_channels or [64, 128, 256]
        layers: List[nn.Module] = []
        in_channels = input_features
        for out_channels in cnn_channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(cnn_dropout),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.fc_classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        if use_regression_head:
            self.fc_regression = nn.Sequential(
                nn.Linear(lstm_hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        last_hidden = self.lstm_dropout(h_n[-1])
        logits = self.fc_classifier(last_hidden)

        magnitude = None
        if self.use_regression_head:
            magnitude = self.fc_regression(last_hidden)
        return logits, magnitude


class TechnicalAgent:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.predictor_root = repo_root / "bitcoin-predictor-dev"
        self.trade_model_path = self.predictor_root / "src" / "models" / "trade" / "best_model.pt"
        self.direction_model_path = self.predictor_root / "src" / "models" / "direction" / "best_model.pt"
        self.features_path = self.predictor_root / "data" / "processed" / "features_1h.parquet"

    @staticmethod
    def _select_features(df: pd.DataFrame) -> List[str]:
        exclude = {
            "open", "high", "low", "close", "volume",
            "label", "hit_bars", "market_return", "trade_return",
            "tp_price", "entry_price", "sl_price", "atr_used", "reward_risk",
        }
        return [c for c in df.columns if c not in exclude]

    @staticmethod
    def _build_model_from_checkpoint(checkpoint_path: Path) -> Tuple[CNNLSTMModel, Dict, int, int]:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]
        sequence_length = int(ckpt.get("config", {}).get("sequence_length", 60))

        input_features = int(state_dict["cnn.0.weight"].shape[1])
        num_classes = int(state_dict["fc_classifier.3.weight"].shape[0])
        use_regression_head = any(key.startswith("fc_regression") for key in state_dict.keys())

        model = CNNLSTMModel(
            input_features=input_features,
            num_classes=num_classes,
            use_regression_head=use_regression_head,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model, ckpt, input_features, sequence_length

    def run(self, trade_threshold: float = 0.55) -> TechnicalSignal:
        trade_model, _, trade_input_features, seq_len = self._build_model_from_checkpoint(self.trade_model_path)
        direction_model, _, dir_input_features, _ = self._build_model_from_checkpoint(self.direction_model_path)

        if trade_input_features != dir_input_features:
            raise ValueError("Trade and direction models expect different feature counts.")

        features_df = pd.read_parquet(self.features_path)
        feature_cols = self._select_features(features_df)

        if len(feature_cols) != trade_input_features:
            raise ValueError(
                f"Feature mismatch: expected {trade_input_features}, got {len(feature_cols)} after selection."
            )

        if len(features_df) < seq_len:
            raise ValueError(f"Not enough rows in features file. Need at least {seq_len} rows.")

        latest_window = features_df[feature_cols].tail(seq_len).to_numpy(dtype="float32")
        x = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            trade_logits, _ = trade_model(x)
            direction_logits, _ = direction_model(x)

        trade_probs = torch.softmax(trade_logits, dim=-1)[0]
        dir_probs = torch.softmax(direction_logits, dim=-1)[0]

        trade_prob = float(trade_probs[1].item())
        long_prob = float(dir_probs[1].item())

        if trade_prob < trade_threshold:
            signal = "hold"
            confidence = max(0.0, min(1.0, 1.0 - trade_prob))
            reasoning = (
                f"Trade model probability is {trade_prob:.2f}, below threshold {trade_threshold:.2f}; "
                f"technical signal stays HOLD."
            )
        else:
            signal = "buy" if long_prob >= 0.5 else "sell"
            direction_conf = max(long_prob, 1.0 - long_prob)
            confidence = max(0.0, min(1.0, trade_prob * direction_conf))
            reasoning = (
                f"Trade model probability is {trade_prob:.2f} (above threshold {trade_threshold:.2f}); "
                f"direction model {'LONG' if signal == 'buy' else 'SHORT'} probability is {direction_conf:.2f}."
            )

        return TechnicalSignal(
            signal=signal,
            confidence=confidence,
            trade_probability=trade_prob,
            long_probability=long_prob,
            sequence_length=seq_len,
            feature_count=len(feature_cols),
            reasoning=reasoning,
        )



