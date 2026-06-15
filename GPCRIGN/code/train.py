#!/usr/bin/env python3
"""Train and evaluate the GPCRIGN binary classification model."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from torch.utils.data import DataLoader

from graph_constructor import GraphDatasetV2MulPro, collate_fn_v2_MulPro
from model import DTIPredictorV4_V2
from utils import set_random_seed


DEFAULT_DATA_ROOT = Path("GPCR_Dataset_Organized24")
DEFAULT_LABEL_CSV = Path("label.csv")
DEFAULT_MODEL_DIR = Path("model_save")
DEFAULT_RESULT_DIR = Path("train_results")

DATASET_FOLDERS = {
    "Train": "Final_Train",
    "Warm_Start": "Warm_Start",
    "Cold_Protein": "Cold_Protein",
    "Cold_Ligand": "Cold_Ligand",
    "Double_Cold": "Double_Cold",
}


class EarlyStopper:
    """Save the best model and stop after a configurable number of stale epochs."""

    def __init__(self, checkpoint_path: Path, patience: int, tolerance: float) -> None:
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.tolerance = tolerance
        self.best_score: float | None = None
        self.stale_epochs = 0

    def step(self, score: float, model: torch.nn.Module) -> bool:
        """Update early-stopping state and return whether training should stop."""
        if not np.isfinite(score):
            raise ValueError("Early-stopping metric must be finite.")

        improved = self.best_score is None or score > self.best_score + self.tolerance
        if improved:
            self.best_score = score
            self.stale_epochs = 0
            ensure_parent(self.checkpoint_path)
            torch.save({"model_state_dict": model.state_dict()}, self.checkpoint_path)
            print(f"[Checkpoint] Saved new best model: {self.checkpoint_path}")
            return False

        self.stale_epochs += 1
        print(f"[Early Stop] Stale epochs: {self.stale_epochs}/{self.patience}")
        return self.stale_epochs >= self.patience


def backup_and_recreate(directory: Path) -> None:
    """Back up an existing cache directory and create an empty replacement."""
    if directory.exists():
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = directory.with_name(f"{directory.name}.bak_{timestamp}")
        shutil.move(str(directory), str(backup))
        print(f"[Cache] Backed up {directory} to {backup}")
    directory.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    """Create a file's parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_logits(model_output: Any) -> torch.Tensor:
    """Extract logits from models that optionally return additional outputs."""
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def load_label_dict(label_csv: Path | None, required: bool = True) -> dict[str, int]:
    """Load labels from a CSV containing `ligand` and `label` columns."""
    if label_csv is None or not label_csv.is_file():
        if required:
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        print(f"[Labels] No optional label file found at {label_csv}.")
        return {}

    frame = pd.read_csv(label_csv)
    required_columns = {"ligand", "label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Label CSV is missing required columns: {missing}")

    label_dict = {
        str(ligand): int(label)
        for ligand, label in zip(frame["ligand"], frame["label"])
    }
    invalid_labels = sorted(set(label_dict.values()).difference({-1, 0, 1}))
    if invalid_labels:
        raise ValueError(f"Label CSV contains unsupported labels: {invalid_labels}")
    return label_dict


def list_sample_files(complex_dir: Path) -> list[str]:
    """Return sorted sample file names from a complex directory."""
    if not complex_dir.is_dir():
        raise FileNotFoundError(f"Complex directory not found: {complex_dir}")

    keys = sorted(
        path.name
        for path in complex_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    if not keys:
        raise ValueError(f"No sample files found in complex directory: {complex_dir}")
    return keys


def create_dataset(
    complex_dir: Path,
    graph_ls_dir: Path,
    graph_dic_dir: Path,
    label_dict: dict[str, int],
    dis_threshold: float,
    num_processes: int,
    limit: int | None = None,
    allow_unlabeled: bool = False,
) -> GraphDatasetV2MulPro:
    """Create a graph dataset and rebuild an invalid cache once."""
    keys = list_sample_files(complex_dir)
    if not allow_unlabeled:
        keys = [key for key in keys if label_dict.get(key) in (0, 1)]
    if limit is not None:
        keys = keys[:limit]
    if not keys:
        raise ValueError(f"No usable labeled samples found in: {complex_dir}")

    labels = [label_dict.get(key, -1) for key in keys]
    data_paths = [str(complex_dir / key) for key in keys]
    graph_ls_dir.mkdir(parents=True, exist_ok=True)
    graph_dic_dir.mkdir(parents=True, exist_ok=True)

    def build() -> GraphDatasetV2MulPro:
        return GraphDatasetV2MulPro(
            keys=keys,
            labels=labels,
            data_dirs=data_paths,
            graph_ls_path=str(graph_ls_dir),
            graph_dic_path=str(graph_dic_dir),
            num_process=num_processes,
            path_marker=os.sep,
            dis_threshold=dis_threshold,
        )

    dataset = build()
    if len(dataset) == 0:
        print(f"[Cache] Empty dataset detected for {complex_dir}; rebuilding once.")
        backup_and_recreate(graph_ls_dir)
        backup_and_recreate(graph_dic_dir)
        dataset = build()

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is still empty after rebuilding cache: {complex_dir}")
    return dataset


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return the mean batch loss."""
    model.train()
    losses: list[float] = []

    for bg, bg3, bg2, bg4, labels, _ in dataloader:
        optimizer.zero_grad()
        bg = bg.to(device)
        bg3 = bg3.to(device)
        bg2 = bg2.to(device)
        bg4 = bg4.to(device)
        labels = labels.to(device)

        logits = extract_logits(model(bg, bg3, bg2, bg4))
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    if not losses:
        raise RuntimeError("Training dataloader produced no batches.")
    return float(np.mean(losses))


def evaluate(
    model: torch.nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float | None, np.ndarray, np.ndarray, list[str]]:
    """Evaluate a dataset once and return loss, labels, probabilities, and keys."""
    model.eval()
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_keys: list[str] = []

    with torch.no_grad():
        for bg, bg3, bg2, bg4, labels, keys in dataloader:
            bg = bg.to(device)
            bg3 = bg3.to(device)
            bg2 = bg2.to(device)
            bg4 = bg4.to(device)
            labels = labels.to(device)

            logits = extract_logits(model(bg, bg3, bg2, bg4))
            labeled_mask = labels != -1
            if labeled_mask.any():
                losses.append(float(loss_fn(logits[labeled_mask], labels[labeled_mask]).item()))

            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_keys.extend(str(key) for key in keys)

    if not all_labels:
        raise RuntimeError("Evaluation dataloader produced no batches.")

    mean_loss = float(np.mean(losses)) if losses else None
    return (
        mean_loss,
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        all_keys,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    """Compute binary classification metrics while ignoring label -1."""
    labeled_mask = y_true != -1
    labeled_count = int(labeled_mask.sum())
    metrics: dict[str, float | int | None] = {
        "N_total": int(len(y_true)),
        "N_labeled": labeled_count,
        "ACC": None,
        "MCC": None,
        "TN": None,
        "FP": None,
        "FN": None,
        "TP": None,
    }
    if labeled_count == 0:
        return metrics

    labeled_true = y_true[labeled_mask]
    labeled_pred = y_pred[labeled_mask]
    tn, fp, fn, tp = confusion_matrix(
        labeled_true,
        labeled_pred,
        labels=[0, 1],
    ).ravel()
    metrics.update(
        {
            "ACC": float(accuracy_score(labeled_true, labeled_pred)),
            "MCC": float(matthews_corrcoef(labeled_true, labeled_pred)),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    )
    return metrics


def export_predictions(
    output_csv: Path,
    keys: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Write per-sample predictions and probabilities to CSV."""
    ensure_parent(output_csv)
    y_pred = probabilities.argmax(axis=1)
    pd.DataFrame(
        {
            "key": keys,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_agonist_0": probabilities[:, 0],
            "prob_antagonist_1": probabilities[:, 1],
        }
    ).to_csv(output_csv, index=False)
    print(f"[Output] Saved predictions: {output_csv}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a raw state dict or a wrapped checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    return model


def create_data_loaders(
    args: argparse.Namespace,
    label_dict: dict[str, int],
    extra_label_dict: dict[str, int],
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Build training and evaluation datasets and dataloaders."""
    datasets: dict[str, GraphDatasetV2MulPro] = {}
    for dataset_name, folder_name in DATASET_FOLDERS.items():
        base_dir = args.data_root / folder_name
        print(f"[Data] Preparing {dataset_name} from {base_dir}")
        datasets[dataset_name] = create_dataset(
            complex_dir=base_dir / "complex",
            graph_ls_dir=base_dir / "graph_ls_path",
            graph_dic_dir=base_dir / "graph_dic_path",
            label_dict=label_dict,
            dis_threshold=args.distance_threshold,
            num_processes=args.num_processes,
            limit=args.limit,
        )
        print(f"[Data] {dataset_name} size: {len(datasets[dataset_name])}")

    if args.extra_complex_dir is not None:
        extra_root = args.extra_complex_dir.parent
        extra_graph_ls_dir = args.extra_graph_ls_dir or extra_root / "graph_ls_path"
        extra_graph_dic_dir = args.extra_graph_dic_dir or extra_root / "graph_dic_path"
        datasets["Extra_Test"] = create_dataset(
            complex_dir=args.extra_complex_dir,
            graph_ls_dir=extra_graph_ls_dir,
            graph_dic_dir=extra_graph_dic_dir,
            label_dict=extra_label_dict,
            dis_threshold=args.distance_threshold,
            num_processes=args.num_processes,
            limit=args.limit,
            allow_unlabeled=True,
        )
        print(f"[Data] Extra_Test size: {len(datasets['Extra_Test'])}")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn_v2_MulPro,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(datasets.pop("Train"), shuffle=True, **loader_options)
    evaluation_loaders = {
        name: DataLoader(dataset, shuffle=False, **loader_options)
        for name, dataset in datasets.items()
    }
    return train_loader, evaluation_loaders


def print_dataset_result(
    dataset_name: str,
    loss: float | None,
    metrics: dict[str, float | int | None],
) -> None:
    """Print one compact evaluation result."""
    loss_text = f"{loss:.4f}" if loss is not None else "NA"
    if metrics["N_labeled"] == 0:
        print(
            f"[{dataset_name:12}] loss={loss_text} | no matched labels "
            f"| samples={metrics['N_total']}"
        )
        return

    print(
        f"[{dataset_name:12}] loss={loss_text} | MCC={metrics['MCC']:.4f} "
        f"| ACC={metrics['ACC']:.4f} | labeled={metrics['N_labeled']}/{metrics['N_total']}"
    )
    print(
        f"{'':14}TN={metrics['TN']} FP={metrics['FP']} "
        f"FN={metrics['FN']} TP={metrics['TP']}"
    )


def save_plots(
    result_dir: Path,
    final_results: list[dict[str, Any]],
    history: pd.DataFrame,
) -> None:
    """Save accuracy and loss plots when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib is not installed; skipping plots.")
        return

    accuracy_frame = pd.DataFrame(final_results).dropna(subset=["ACC"])
    if not accuracy_frame.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(accuracy_frame["Dataset"], accuracy_frame["ACC"].astype(float))
        plt.title("Accuracy Across Labeled Datasets")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=20)
        plt.tight_layout()
        accuracy_plot = result_dir / "accuracy_comparison.png"
        plt.savefig(accuracy_plot, dpi=300)
        plt.close()
        print(f"[Plot] Saved accuracy plot: {accuracy_plot}")

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    for column in history.columns:
        if not column.endswith("_loss") or column == "train_loss":
            continue
        values = pd.to_numeric(history[column], errors="coerce")
        if np.isfinite(values).any():
            plt.plot(history["epoch"], values, label=column)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot = result_dir / "loss_curves.png"
    plt.savefig(loss_plot, dpi=300)
    plt.close()
    print(f"[Plot] Saved loss plot: {loss_plot}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpuid", default="0", help="CUDA device index.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", "--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)
    parser.add_argument("--weight-decay", "--l2", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--num-processes", "--num_process", type=int, default=6)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset sample limit.",
    )

    model_group = parser.add_argument_group("model architecture")
    model_group.add_argument("--node-feat-size", "--node_feat_size", type=int, default=40)
    model_group.add_argument("--edge-feat-size-3d", "--edge_feat_size_3d", type=int, default=21)
    model_group.add_argument("--graph-feat-size", "--graph_feat_size", type=int, default=256)
    model_group.add_argument("--num-layers", "--num_layers", type=int, default=4)
    model_group.add_argument("--outdim-g3", "--outdim_g3", type=int, default=256)
    model_group.add_argument("--fc-hidden-size", "--d_FC_layer", type=int, default=512)
    model_group.add_argument("--fc-layers", "--n_FC_layer", type=int, default=4)
    model_group.add_argument("--dropout", type=float, default=0.35)
    model_group.add_argument("--num-tasks", "--n_tasks", type=int, default=2)
    model_group.add_argument("--distance-threshold", "--dis_threshold", type=float, default=5.0)

    path_group = parser.add_argument_group("paths")
    path_group.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    path_group.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    path_group.add_argument(
        "--model-save-dir",
        "--model_save_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    path_group.add_argument("--checkpoint-name", default="best_model.pth")
    path_group.add_argument("--result-dir", "--train_result", type=Path, default=DEFAULT_RESULT_DIR)

    extra_group = parser.add_argument_group("optional extra test set")
    extra_group.add_argument("--extra-complex-dir", "--extra_complex_dir", type=Path, default=None)
    extra_group.add_argument("--extra-graph-ls-dir", "--extra_graph_ls", type=Path, default=None)
    extra_group.add_argument("--extra-graph-dic-dir", "--extra_graph_dic", type=Path, default=None)
    extra_group.add_argument("--extra-label-csv", "--extra_label_csv", type=Path, default=None)
    extra_group.add_argument("--extra-output-csv", "--extra_out_csv", type=Path, default=None)
    return parser


def main() -> None:
    """Train the model from command-line arguments."""
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    args.model_save_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    label_dict = load_label_dict(args.label_csv)
    extra_label_dict = (
        load_label_dict(args.extra_label_csv, required=False)
        if args.extra_complex_dir is not None
        else {}
    )
    train_loader, evaluation_loaders = create_data_loaders(args, label_dict, extra_label_dict)

    train_labels = [
        int(label)
        for label in train_loader.dataset.labels
        if int(label) in (0, 1)
    ]
    negative_count = train_labels.count(0)
    positive_count = train_labels.count(1)
    if negative_count == 0 or positive_count == 0:
        raise ValueError("Training data must contain both class 0 and class 1.")

    class_weights = torch.tensor(
        [
            len(train_labels) / (2.0 * negative_count),
            len(train_labels) / (2.0 * positive_count),
        ],
        dtype=torch.float32,
    )
    print(f"[Train] Class weights: {class_weights.tolist()}")

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    model = DTIPredictorV4_V2(
        node_feat_size=args.node_feat_size,
        edge_feat_size=args.edge_feat_size_3d,
        num_layers=args.num_layers,
        graph_feat_size=args.graph_feat_size,
        outdim_g3=args.outdim_g3,
        d_FC_layer=args.fc_hidden_size,
        n_FC_layer=args.fc_layers,
        dropout=args.dropout,
        n_tasks=args.num_tasks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.95, 0.999),
        amsgrad=True,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    checkpoint_path = args.model_save_dir / args.checkpoint_name
    early_stopper = EarlyStopper(checkpoint_path, args.patience, args.tolerance)
    history_rows: list[dict[str, float | int | None]] = []
    history_csv = args.result_dir / "loss_history.csv"

    print("[Train] Starting training.")
    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        history_row: dict[str, float | int | None] = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        epoch_metrics: dict[str, dict[str, float | int | None]] = {}

        print(f"\n[Epoch {epoch}/{args.epochs}]")
        for dataset_name, dataloader in evaluation_loaders.items():
            eval_loss, y_true, probabilities, _ = evaluate(model, loss_fn, dataloader, device)
            metrics = compute_metrics(y_true, probabilities.argmax(axis=1))
            epoch_metrics[dataset_name] = metrics
            history_row[f"{dataset_name}_loss"] = eval_loss
            print_dataset_result(dataset_name, eval_loss, metrics)

        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(history_csv, index=False)
        print(f"[Epoch {epoch}] Completed in {time.time() - started_at:.2f}s")

        cold_ligand_mcc = epoch_metrics["Cold_Ligand"]["MCC"]
        if cold_ligand_mcc is None:
            raise ValueError("Cold_Ligand has no labels; early stopping cannot be evaluated.")
        if early_stopper.step(float(cold_ligand_mcc), model):
            print(f"[Early Stop] Stopping after epoch {epoch}.")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = load_checkpoint(model, checkpoint_path, device)
    print(f"\n[Final] Loaded best checkpoint: {checkpoint_path}")
    final_results: list[dict[str, Any]] = []

    for dataset_name, dataloader in evaluation_loaders.items():
        eval_loss, y_true, probabilities, keys = evaluate(model, loss_fn, dataloader, device)
        metrics = compute_metrics(y_true, probabilities.argmax(axis=1))
        final_results.append({"Dataset": dataset_name, "Loss": eval_loss, **metrics})
        print_dataset_result(dataset_name, eval_loss, metrics)

        if dataset_name == "Extra_Test":
            extra_output_csv = (
                args.extra_output_csv
                or args.result_dir / "extra_test_predictions.csv"
            )
            export_predictions(extra_output_csv, keys, y_true, probabilities)

    summary_csv = args.result_dir / "final_performance_summary.csv"
    pd.DataFrame(final_results).to_csv(summary_csv, index=False)
    print(f"[Output] Saved final summary: {summary_csv}")

    history_frame = pd.DataFrame(history_rows)
    save_plots(args.result_dir, final_results, history_frame)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
