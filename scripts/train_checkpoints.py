#!/usr/bin/env python3
"""Train univariate DL checkpoints matching CastMind inference hyperparameters."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from castmind.DeepLearningModels.Autoformer import Model as Autoformer
from castmind.DeepLearningModels.DLinear import Model as DLinear
from castmind.DeepLearningModels.PatchTST import Model as PatchTST
from castmind.DeepLearningModels.TimesNet import Model as TimesNet
from castmind.DeepLearningModels.iTransformer import Model as iTransformer
from castmind.utils.timefeatures import time_features

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "castmind" / "DeepLearningCheckpoints"

MODELS = {
    "DLinear": DLinear,
    "TimesNet": TimesNet,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "Autoformer": Autoformer,
}


class WindowDataset(Dataset):
    def __init__(self, y: np.ndarray, stamps: np.ndarray, seq_len: int, pred_len: int, label_len: int, stride: int = 1):
        self.y = y.astype(np.float32)
        self.stamps = stamps.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.starts = list(range(0, max(0, len(y) - seq_len - pred_len + 1), max(1, stride)))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.seq_len
        p = e + self.pred_len
        x = self.y[s:e]
        y = self.y[e:p]
        x_mark = self.stamps[s:e]
        y_mark = self.stamps[e - self.label_len : p]
        x_dec = np.concatenate([self.y[e - self.label_len : e], np.zeros(self.pred_len, dtype=np.float32)])
        return (
            torch.from_numpy(x[:, None]),
            torch.from_numpy(x_mark),
            torch.from_numpy(x_dec[:, None]),
            torch.from_numpy(y_mark),
            torch.from_numpy(y[:, None]),
        )


def make_args(model_name: str, seq_len: int, pred_len: int) -> SimpleNamespace:
    args = SimpleNamespace()
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.model = model_name
    args.freq = "h"
    args.seq_len = seq_len
    args.label_len = min(48, seq_len)
    args.pred_len = pred_len
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    args.features = "S"
    args.embed = "timeF"
    args.activation = "gelu"
    args.output_attention = False
    args.dropout = 0.1
    args.moving_avg = 25
    args.factor = 3
    args.distil = True
    args.top_k = 5
    args.num_kernels = 6
    args.individual = 0
    if model_name == "TimesNet":
        args.d_model = 16
        args.d_ff = 32
        args.e_layers = 2
        args.n_heads = 8
    elif model_name == "iTransformer":
        args.d_model = 128
        args.d_ff = 128
        args.e_layers = 2
        args.n_heads = 8
    elif model_name == "PatchTST":
        args.d_model = 64
        args.d_ff = 128
        args.e_layers = 1
        args.n_heads = 2
        args.patch_len = 16
    else:
        args.d_model = 64
        args.d_ff = 128
        args.e_layers = 2
        args.d_layers = 1
        args.n_heads = 8
    args.d_layers = getattr(args, "d_layers", 1)
    return args


def load_series(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    target = df.columns[-1]
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(y)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]
    mean, std = float(np.mean(y)), float(np.std(y) + 1e-6)
    y = (y - mean) / std
    stamp = time_features(pd.to_datetime(df["date"].values), freq="h").transpose(1, 0)
    return y, stamp


def train_one(model_cls, args, loader: DataLoader, device: torch.device, epochs: int) -> torch.nn.Module:
    model = model_cls(args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        losses: List[float] = []
        for x_enc, x_mark, x_dec, y_mark, y in loader:
            x_enc, x_mark, x_dec, y_mark, y = [t.to(device) for t in (x_enc, x_mark, x_dec, y_mark, y)]
            optim.zero_grad()
            out = model(x_enc, x_mark, x_dec, y_mark)
            if isinstance(out, tuple):
                out = out[0]
            pred = out[:, -args.pred_len :, :1]
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        print(f"    epoch {epoch + 1}/{epochs} mse={np.mean(losses):.5f}")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["DLinear", "TimesNet", "PatchTST", "iTransformer", "Autoformer"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args_cli = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[info] training on {device}")

    datasets: Dict[str, Tuple[Path, int, int]] = {
        "ETTh1": (ROOT / "data/ETTh1/train.csv", 96, 96),
        "ETTm1": (ROOT / "data/ETTm1/train.csv", 96, 96),
        "EPF_NP": (ROOT / "data/EPF_NP/train.csv", 168, 24),
        "EPF_PJM": (ROOT / "data/EPF_PJM/train.csv", 168, 24),
        "EPF_BE": (ROOT / "data/EPF_BE/train.csv", 168, 24),
        "EPF_FR": (ROOT / "data/EPF_FR/train.csv", 168, 24),
        "EPF_DE": (ROOT / "data/EPF_DE/train.csv", 168, 24),
        "windy_power": (ROOT / "data/windy_power/train.csv", 96, 96),
        "sunny_power": (ROOT / "data/sunny_power/train.csv", 96, 96),
        "MOPEX": (ROOT / "data/MOPEX/train.csv", 96, 96),
    }

    for ds_name, (csv_path, seq_len, pred_len) in datasets.items():
        if not csv_path.exists():
            print(f"[skip] missing {csv_path}")
            continue
        print(f"\n=== {ds_name} seq={seq_len} pred={pred_len} ===")
        y, stamps = load_series(csv_path)
        label_len = min(48, seq_len)
        out_dir = CKPT_ROOT / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for model_name in args_cli.models:
            if model_name not in MODELS:
                continue
            ckpt_path = out_dir / f"{model_name}.pth"
            if ckpt_path.exists():
                print(f"  [skip] {ckpt_path.name} exists")
                continue
            stride = 1 if model_name == "DLinear" else 8
            dataset = WindowDataset(y, stamps, seq_len, pred_len, label_len, stride=stride)
            if len(dataset) == 0:
                print("[skip] not enough rows")
                continue
            print(f"  training {model_name} windows={len(dataset)} stride={stride} -> {ckpt_path}")
            loader = DataLoader(dataset, batch_size=args_cli.batch_size, shuffle=True, drop_last=False)
            model_args = make_args(model_name, seq_len, pred_len)
            train_device = torch.device("cpu") if model_name == "Autoformer" else device
            try:
                model = train_one(MODELS[model_name], model_args, loader, train_device, args_cli.epochs)
                torch.save(model.state_dict(), ckpt_path)
            except Exception as exc:
                print(f"  [warn] {model_name} failed: {exc}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
