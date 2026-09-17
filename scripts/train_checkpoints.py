#!/usr/bin/env python3
"""Optional local DL checkpoint trainer (NOT the author / TSLib official script).

Protocol (AlphaCast Table 5/6, aligned with data/ after 2026-09-16):
  - Fit on data/<ds>/train.csv only
  - Select / early-stop by Val MSE on data/<ds>/val.csv
  - Never reads test.csv

Prefer TSLib official-split weights when you already have them under
castmind/DeepLearningCheckpoints/<ds>/. Use this script only to fill gaps or
to retrain with the same CastMind inference hyperparams (freq='t' marks).

Examples:
  bash scripts/run.sh scripts/train_checkpoints.py --datasets ETTh1 --force
  bash scripts/run.sh scripts/train_checkpoints.py --preset light --patience 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from castmind.DeepLearningModels.Autoformer import Model as Autoformer
from castmind.DeepLearningModels.DLinear import Model as DLinear
from castmind.DeepLearningModels.PatchTST import Model as PatchTST
from castmind.DeepLearningModels.TimesNet import Model as TimesNet
from castmind.DeepLearningModels.iTransformer import Model as iTransformer
from castmind.DeepLearningModels.TimeXer import Model as TimeXer
from castmind.utils.timefeatures import time_features

CKPT_ROOT = ROOT / "castmind" / "DeepLearningCheckpoints"

# Match config.yaml default pool (TimeXer opt-in via --models).
MODELS = {
    "DLinear": DLinear,
    "TimesNet": TimesNet,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "Autoformer": Autoformer,
    "TimeXer": TimeXer,
}
DEFAULT_MODELS = ["DLinear", "TimesNet", "PatchTST", "iTransformer", "Autoformer"]

# timefeat freq kept as 't' so marks dim matches CastMind inference / TSLib-guide ckpts.
TIMEFEAT_FREQ = "t"


class WindowDataset(Dataset):
    def __init__(
        self,
        y: np.ndarray,
        stamps: np.ndarray,
        seq_len: int,
        pred_len: int,
        label_len: int,
        stride: int = 1,
    ):
        self.y = y.astype(np.float32)
        self.stamps = stamps.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.starts = list(
            range(0, max(0, len(y) - seq_len - pred_len + 1), max(1, stride))
        )

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
        x_dec = np.concatenate(
            [self.y[e - self.label_len : e], np.zeros(self.pred_len, dtype=np.float32)]
        )
        return (
            torch.from_numpy(x[:, None]),
            torch.from_numpy(x_mark),
            torch.from_numpy(x_dec[:, None]),
            torch.from_numpy(y_mark),
            torch.from_numpy(y[:, None]),
        )


def make_args(model_name: str, seq_len: int, pred_len: int, preset: str = "light") -> SimpleNamespace:
    args = SimpleNamespace()
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.model = model_name
    args.freq = TIMEFEAT_FREQ
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
    from castmind.models.base import dl_backbone_hparams

    bb = dl_backbone_hparams(model_name, preset=preset)
    args.d_model = bb["d_model"]
    args.d_ff = bb["d_ff"]
    args.e_layers = bb["e_layers"]
    args.n_heads = bb["n_heads"]
    args.d_layers = bb["d_layers"]
    if model_name == "PatchTST":
        args.patch_len = 16
    if model_name == "TimeXer":
        args.patch_len = 16
        args.use_norm = 1
        args.features = "M"
    return args


def load_series(
    csv_path: Path,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Load target (last column). Fit mean/std on train; reuse for val."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    target = df.columns[-1]
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(y)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]
    if mean is None or std is None:
        mean = float(np.mean(y))
        std = float(np.std(y) + 1e-6)
    y = (y - mean) / std
    stamp = time_features(pd.to_datetime(df["date"].values), freq=TIMEFEAT_FREQ).transpose(1, 0)
    return y, stamp, float(mean), float(std)


@torch.no_grad()
def eval_mse(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, pred_len: int
) -> float:
    model.eval()
    losses: List[float] = []
    for x_enc, x_mark, x_dec, y_mark, y in loader:
        x_enc, x_mark, x_dec, y_mark, y = [
            t.to(device) for t in (x_enc, x_mark, x_dec, y_mark, y)
        ]
        out = model(x_enc, x_mark, x_dec, y_mark)
        if isinstance(out, tuple):
            out = out[0]
        pred = out[:, -pred_len:, :1]
        losses.append(float(torch.mean((pred - y) ** 2).item()))
    return float(np.mean(losses)) if losses else float("inf")


def train_one(
    model_cls,
    args,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    patience: int = 3,
) -> Tuple[torch.nn.Module, dict]:
    """Fit on Train; keep state_dict with best Val MSE (early stop)."""
    model = model_cls(args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for x_enc, x_mark, x_dec, y_mark, y in train_loader:
            x_enc, x_mark, x_dec, y_mark, y = [
                t.to(device) for t in (x_enc, x_mark, x_dec, y_mark, y)
            ]
            optim.zero_grad()
            out = model(x_enc, x_mark, x_dec, y_mark)
            if isinstance(out, tuple):
                out = out[0]
            pred = out[:, -args.pred_len :, :1]
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        train_mse = float(np.mean(losses)) if losses else float("inf")

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_mse = eval_mse(model, val_loader, device, args.pred_len)
            print(
                f"    epoch {epoch}/{epochs} train_mse={train_mse:.5f} val_mse={val_mse:.5f}"
            )
            history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
            if val_mse < best_val - 1e-8:
                best_val = val_mse
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(
                        f"    early stop at epoch {epoch} "
                        f"(best epoch={best_epoch} val_mse={best_val:.5f})"
                    )
                    break
        else:
            print(f"    epoch {epoch}/{epochs} train_mse={train_mse:.5f} (no val)")
            history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": None})
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    meta = {
        "best_epoch": best_epoch,
        "best_val_mse": None if best_val == float("inf") else best_val,
        "epochs_run": len(history),
        "history": history,
    }
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max epochs (default: 3 light / 10 official)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many epochs without Val improvement",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--preset",
        choices=["light", "official"],
        default="light",
        help="light → DeepLearningCheckpoints/<ds>/; official → <ds>_official/",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing .pth")
    parser.add_argument(
        "--allow-no-val",
        action="store_true",
        help="Allow missing val.csv (discouraged; no early-stop)",
    )
    args_cli = parser.parse_args()
    preset = args_cli.preset
    epochs = args_cli.epochs if args_cli.epochs is not None else (10 if preset == "official" else 3)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(
        f"[info] train=Train.csv fit | val=Val.csv early-stop | "
        f"device={device} preset={preset} epochs={epochs} patience={args_cli.patience}"
    )

    # (train_csv, val_csv, seq_len, pred_len) — paper short 168/24, long 96/96
    datasets: Dict[str, Tuple[Path, Path, int, int]] = {
        "ETTh1": (ROOT / "data/ETTh1/train.csv", ROOT / "data/ETTh1/val.csv", 96, 96),
        "ETTm1": (ROOT / "data/ETTm1/train.csv", ROOT / "data/ETTm1/val.csv", 96, 96),
        "EPF_NP": (ROOT / "data/EPF_NP/train.csv", ROOT / "data/EPF_NP/val.csv", 168, 24),
        "EPF_PJM": (ROOT / "data/EPF_PJM/train.csv", ROOT / "data/EPF_PJM/val.csv", 168, 24),
        "EPF_BE": (ROOT / "data/EPF_BE/train.csv", ROOT / "data/EPF_BE/val.csv", 168, 24),
        "EPF_FR": (ROOT / "data/EPF_FR/train.csv", ROOT / "data/EPF_FR/val.csv", 168, 24),
        "EPF_DE": (ROOT / "data/EPF_DE/train.csv", ROOT / "data/EPF_DE/val.csv", 168, 24),
        "windy_power": (
            ROOT / "data/windy_power/train.csv",
            ROOT / "data/windy_power/val.csv",
            96,
            96,
        ),
        "sunny_power": (
            ROOT / "data/sunny_power/train.csv",
            ROOT / "data/sunny_power/val.csv",
            96,
            96,
        ),
        "MOPEX": (ROOT / "data/MOPEX/train.csv", ROOT / "data/MOPEX/val.csv", 96, 96),
    }
    if args_cli.datasets:
        datasets = {k: v for k, v in datasets.items() if k in set(args_cli.datasets)}

    for ds_name, (train_csv, val_csv, seq_len, pred_len) in datasets.items():
        if not train_csv.exists():
            print(f"[skip] missing {train_csv}")
            continue
        if not val_csv.exists() and not args_cli.allow_no_val:
            print(f"[skip] missing {val_csv} (pass --allow-no-val to train without Val)")
            continue

        print(f"\n=== {ds_name} seq={seq_len} pred={pred_len} preset={preset} ===")
        print(f"  train={train_csv}")
        print(f"  val  ={val_csv if val_csv.exists() else '(none)'}")

        y_tr, stamps_tr, mean, std = load_series(train_csv)
        label_len = min(48, seq_len)
        out_dir = CKPT_ROOT / (f"{ds_name}_official" if preset == "official" else ds_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        if val_csv.exists():
            y_va, stamps_va, _, _ = load_series(val_csv, mean=mean, std=std)
        else:
            print("  [warn] training without Val early-stop")
            y_va, stamps_va = None, None

        run_meta: dict = {
            "dataset": ds_name,
            "preset": preset,
            "train_csv": str(train_csv.relative_to(ROOT)),
            "val_csv": str(val_csv.relative_to(ROOT)) if val_csv.exists() else None,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "normalize_mean": mean,
            "normalize_std": std,
            "timefeat_freq": TIMEFEAT_FREQ,
            "patience": args_cli.patience,
            "max_epochs": epochs,
            "note": "Fit on Train only; checkpoint = best Val MSE. Not author Table-1 weights.",
            "models": {},
        }

        for model_name in args_cli.models:
            if model_name not in MODELS:
                print(f"  [skip] unknown model {model_name}")
                continue
            ckpt_path = out_dir / f"{model_name}.pth"
            if ckpt_path.exists() and not args_cli.force:
                print(f"  [skip] {ckpt_path.name} exists (use --force to retrain)")
                continue

            stride = 1 if model_name == "DLinear" else 8
            train_ds = WindowDataset(y_tr, stamps_tr, seq_len, pred_len, label_len, stride=stride)
            if len(train_ds) == 0:
                print("  [skip] not enough train rows")
                continue
            train_loader = DataLoader(
                train_ds, batch_size=args_cli.batch_size, shuffle=True, drop_last=False
            )

            val_loader = None
            if y_va is not None and stamps_va is not None:
                val_ds = WindowDataset(y_va, stamps_va, seq_len, pred_len, label_len, stride=stride)
                if len(val_ds) > 0:
                    val_loader = DataLoader(
                        val_ds, batch_size=args_cli.batch_size, shuffle=False, drop_last=False
                    )

            n_val = len(val_loader.dataset) if val_loader is not None else 0
            print(
                f"  training {model_name} train_windows={len(train_ds)} "
                f"val_windows={n_val} stride={stride} -> {ckpt_path}"
            )
            model_args = make_args(model_name, seq_len, pred_len, preset=preset)
            train_device = torch.device("cpu") if model_name == "Autoformer" else device
            try:
                model, fit_meta = train_one(
                    MODELS[model_name],
                    model_args,
                    train_loader,
                    val_loader,
                    train_device,
                    epochs,
                    patience=args_cli.patience,
                )
                torch.save(model.state_dict(), ckpt_path)
                run_meta["models"][model_name] = fit_meta
                print(
                    f"  [ok] {model_name} best_epoch={fit_meta['best_epoch']} "
                    f"best_val_mse={fit_meta['best_val_mse']}"
                )
            except Exception as exc:
                print(f"  [warn] {model_name} failed: {exc}")

        meta_path = out_dir / "train_checkpoints_meta.json"
        meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        print(f"  [meta] {meta_path}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
