
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class BTCSequenceDataset(Dataset):

    def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            returns: np.ndarray,
            timestamps: pd.DatetimeIndex,
            sequence_length: int = 60,
    ):
        self.features = features
        self.labels = labels
        self.returns = returns
        self.timestamps = timestamps
        self.seq_len = sequence_length

        self.valid_indices = np.arange(sequence_length, len(features))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        X = self.features[actual_idx - self.seq_len: actual_idx]
        y = self.labels[actual_idx]

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor([y])[0]

        metadata = {
            "timestamp": str(self.timestamps[actual_idx]),
            "return": float(self.returns[actual_idx]),
            "original_label": int(y),
        }

        return X_tensor, y_tensor, metadata


def select_features(df: pd.DataFrame) -> Tuple[list, list]:
    exclude = {
        "open", "high", "low", "close", "volume",
        "label", "hit_bars", "market_return", "trade_return",
        "tp_price", "entry_price", "sl_price", "atr_used", "reward_risk",
    }

    feature_cols = [
        col for col in df.columns
        if col not in exclude
    ]

    excluded = [col for col in df.columns if col not in feature_cols]

    return feature_cols, excluded


def temporal_split(
        df: pd.DataFrame,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"\nTemporal Split:")
    print(f"  Train : {len(train):>7,}  |  {train.index[0]}  →  {train.index[-1]}")
    print(f"  Val   : {len(val):>7,}  |  {val.index[0]}  →  {val.index[-1]}")
    print(f"  Test  : {len(test):>7,}  |  {test.index[0]}  →  {test.index[-1]}")

    return train, val, test


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    weights = total / (len(unique) * counts)

    weight_dict = dict(zip(unique, weights))
    weight_tensor = torch.FloatTensor([
        weight_dict.get(0, 1.0),
        weight_dict.get(1, 1.0),
    ])

    print(f"\nClass Weights:")
    print(f"  Class 0 : {weight_tensor[0]:.3f}")
    print(f"  Class 1 : {weight_tensor[1]:.3f}")

    return weight_tensor


def handle_nans(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    n_after = len(df)

    if n_before > n_after:
        print(f"\n⚠️  Dropped {n_before - n_after:,} rows with NaN values")

    return df


def preprocess_data(
        features_file_name: str = "features_1h.parquet",
        labels_file_name: str = "labels_1h.parquet",
        sequence_length: int = 60,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        batch_size: int = 256,
        num_workers: int = 4,
) -> dict:

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    print(f"\n[1/7] Loading features and labels...")
    features_full_path = PROCESSED_DIR / features_file_name
    labels_full_path = PROCESSED_DIR / labels_file_name

    features_df = pd.read_parquet(features_full_path)
    labels_df = pd.read_parquet(labels_full_path)

    common_idx = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    print(f"      Aligned to {len(common_idx):,} rows")

    print(f"\n[2/7] Merging features and labels...")
    df = features_df.join(labels_df, how="inner")
    print(f"      Final shape: {df.shape}")

    print(f"\n[3/7] Selecting features...")
    feature_cols, excluded = select_features(df)
    print(f"      Selected {len(feature_cols)} features")
    print(f"      Excluded {len(excluded)} columns")

    print(f"\n[4/7] Handling NaN values...")
    df = handle_nans(df, feature_cols)

    print(f"\n[5/7] Temporal train/val/test split...")
    train_df, val_df, test_df = temporal_split(df, train_pct, val_pct)

    print(f"\n[6/7] Computing class weights...")
    class_weights = compute_class_weights(train_df["label"].values)
    print(class_weights)

    print(f"\n[7/7] Creating PyTorch datasets...")

    train_dataset = BTCSequenceDataset(
        features=train_df[feature_cols].values,
        labels=train_df["label"].values,
        returns=train_df["trade_return"].values,
        timestamps=train_df.index,
        sequence_length=sequence_length,
    )

    val_dataset = BTCSequenceDataset(
        features=val_df[feature_cols].values,
        labels=val_df["label"].values,
        returns=val_df["trade_return"].values,
        timestamps=val_df.index,
        sequence_length=sequence_length,
    )

    test_dataset = BTCSequenceDataset(
        features=test_df[feature_cols].values,
        labels=test_df["label"].values,
        returns=test_df["trade_return"].values,
        timestamps=test_df.index,
        sequence_length=sequence_length,
    )

    print(f"      Train sequences : {len(train_dataset):,}")
    print(f"      Val sequences   : {len(val_dataset):,}")
    print(f"      Test sequences  : {len(test_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Sequence length : {sequence_length} bars")
    print(f"  Num features    : {len(feature_cols)}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Train batches   : {len(train_loader)}")
    print(f"  Val batches     : {len(val_loader)}")
    print(f"  Test batches    : {len(test_loader)}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
        "feature_cols": feature_cols,
        "sequence_length": sequence_length,
        "num_features": len(feature_cols),
        "metadata": {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "train_date_range": (train_df.index[0], train_df.index[-1]),
            "val_date_range": (val_df.index[0], val_df.index[-1]),
            "test_date_range": (test_df.index[0], test_df.index[-1]),
        }
    }


def inspect_batch(loader: DataLoader, num_batches: int = 1):
    print("\n" + "=" * 60)
    print("BATCH INSPECTION")
    print("=" * 60)

    for i, (X, y, meta) in enumerate(loader):
        if i >= num_batches:
            break

        print(f"\nBatch {i + 1}:")
        print(f"  X shape    : {X.shape}  (batch, seq_len, features)")
        print(f"  y shape    : {y.shape}  (batch,)")
        print(f"  y values   : {y[:10].tolist()}")
        print(f"  y range    : [{y.min().item()}, {y.max().item()}]")
        print(f"  Timestamps : {meta['timestamp'][:3]}")


def save_preprocessing_artifacts(data_dict: dict, output_dir: Path = PROCESSED_DIR):
    import json

    artifact = {
        "feature_cols": data_dict["feature_cols"],
        "sequence_length": data_dict["sequence_length"],
        "num_features": data_dict["num_features"],
        "class_weights": data_dict["class_weights"].tolist(),
        "metadata": {
            k: (str(v[0]), str(v[1])) if isinstance(v, tuple) else v
            for k, v in data_dict["metadata"].items()
        }
    }

    out_path = output_dir / "preprocessing_config.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\n✓  Saved preprocessing config → {out_path}")


if __name__ == "__main__":
    data = preprocess_data(
        sequence_length=60,
        train_pct=0.70,
        val_pct=0.15,
        batch_size=256,
        num_workers=4,
    )

    inspect_batch(data["train_loader"], num_batches=1)

    save_preprocessing_artifacts(data)

    print("\n✓  Ready for model training!")