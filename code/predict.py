#!/usr/bin/env python3
"""Run inference with a trained GPCRIGN model."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from torch.utils.data import DataLoader

from graph_constructor import GraphDatasetV2MulPro, collate_fn_v2_MulPro
from model import DTIPredictorV4_V2
from utils import set_random_seed


DEFAULT_MODEL_PATH = Path("model_save") / "best_model.pth"
DEFAULT_DATA_DIR = Path("data") / "toy_example" / "test"
DEFAULT_COMPLEX_DIR = DEFAULT_DATA_DIR / "complex"
DEFAULT_GRAPH_DIC_DIR = DEFAULT_DATA_DIR / "graph_dic_path"
DEFAULT_GRAPH_LS_DIR = DEFAULT_DATA_DIR / "graph_ls_path"
DEFAULT_LABEL_CSV = Path("data") / "toy_example" / "label.csv"
DEFAULT_OUTPUT_CSV = Path("prediction_results") / "predictions.csv"


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


def load_label_dict(label_csv: Path | None) -> dict[str, int]:
    """Load labels from a CSV containing `ligand` and `label` columns."""
    if label_csv is None or not label_csv.is_file():
        print(f"[Labels] No label file found at {label_csv}; metrics will be skipped.")
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
    force_rebuild_cache: bool = False,
) -> tuple[GraphDatasetV2MulPro, list[str]]:
    """Create an inference dataset and rebuild an invalid cache once."""
    if force_rebuild_cache:
        backup_and_recreate(graph_ls_dir)
        backup_and_recreate(graph_dic_dir)
    else:
        graph_ls_dir.mkdir(parents=True, exist_ok=True)
        graph_dic_dir.mkdir(parents=True, exist_ok=True)

    keys = list_sample_files(complex_dir)
    if limit is not None:
        keys = keys[:limit]
    if not keys:
        raise ValueError("The sample limit produced an empty dataset.")

    labels = [label_dict.get(key, -1) for key in keys]
    data_paths = [str(complex_dir / key) for key in keys]

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

    print(f"[Data] Samples: {len(keys)}")
    print(f"[Data] Labeled samples: {sum(label != -1 for label in labels)}")
    dataset = build()

    if len(dataset) == 0:
        print("[Cache] Empty dataset detected; rebuilding graph cache once.")
        backup_and_recreate(graph_ls_dir)
        backup_and_recreate(graph_dic_dir)
        dataset = build()

    if len(dataset) == 0:
        raise RuntimeError("Dataset is still empty after rebuilding the graph cache.")

    print(f"[Data] Graph dataset size: {len(dataset)}")
    return dataset, keys


def extract_logits(model_output: Any) -> torch.Tensor:
    """Extract logits from models that optionally return additional outputs."""
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a raw state dict or a wrapped checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if state_dict and all(key.startswith("module.") for key in state_dict):
        prefix_length = len("module.")
        state_dict = {
            key[prefix_length:]: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    print(f"[Model] Loaded checkpoint: {checkpoint_path}")
    return model


def run_prediction(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return keys, labels, classes, probabilities, and logits."""
    model.eval()
    all_keys: list[str] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            bg, bg3, bg2, bg4, labels, keys = batch
            bg = bg.to(device)
            bg3 = bg3.to(device)
            bg2 = bg2.to(device)
            bg4 = bg4.to(device)

            logits = extract_logits(model(bg, bg3, bg2, bg4))
            probabilities = torch.softmax(logits, dim=1)

            all_keys.extend(str(key) for key in keys)
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

            if batch_index % 20 == 0:
                print(f"[Predict] Completed batch {batch_index}")

    y_true = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    logits = np.concatenate(all_logits)
    y_pred = probabilities.argmax(axis=1)
    return all_keys, y_true, y_pred, probabilities, logits


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
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
) -> pd.DataFrame:
    """Write per-sample prediction details to CSV."""
    ensure_parent(output_csv)
    class_names = {0: "agonist", 1: "antagonist"}
    frame = pd.DataFrame(
        {
            "key": keys,
            "y_true": y_true,
            "true_name": [class_names.get(int(value), "unlabeled") for value in y_true],
            "y_pred": y_pred,
            "pred_name": [class_names[int(value)] for value in y_pred],
            "prob_agonist_0": probabilities[:, 0],
            "prob_antagonist_1": probabilities[:, 1],
            "logit_agonist_0": logits[:, 0],
            "logit_antagonist_1": logits[:, 1],
        }
    )
    frame.to_csv(output_csv, index=False)
    print(f"[Output] Saved predictions: {output_csv}")
    return frame


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpuid", default="0", help="CUDA device index.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--num-processes", "--num_process", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")

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
    model_group.add_argument("--distance-threshold", "--dis_threshold", type=float, default=6.0)

    path_group = parser.add_argument_group("paths")
    path_group.add_argument("--model-path", "--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    path_group.add_argument(
        "--complex-dir",
        "--complex_dir",
        type=Path,
        default=DEFAULT_COMPLEX_DIR,
    )
    path_group.add_argument(
        "--graph-dic-dir",
        "--graph_dic_path",
        type=Path,
        default=DEFAULT_GRAPH_DIC_DIR,
    )
    path_group.add_argument(
        "--graph-ls-dir",
        "--graph_ls_path",
        type=Path,
        default=DEFAULT_GRAPH_LS_DIR,
    )
    path_group.add_argument("--label-csv", "--label_csv", type=Path, default=DEFAULT_LABEL_CSV)
    path_group.add_argument("--output-csv", "--out_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    path_group.add_argument("--summary-csv", "--summary_csv", type=Path, default=None)

    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Back up and rebuild the graph cache before inference.",
    )
    return parser


def main() -> None:
    """Run prediction from command-line arguments."""
    args = build_parser().parse_args()
    set_random_seed(args.seed)

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    label_dict = load_label_dict(args.label_csv)
    dataset, _ = create_dataset(
        complex_dir=args.complex_dir,
        graph_ls_dir=args.graph_ls_dir,
        graph_dic_dir=args.graph_dic_dir,
        label_dict=label_dict,
        dis_threshold=args.distance_threshold,
        num_processes=args.num_processes,
        limit=args.limit,
        force_rebuild_cache=args.force_rebuild_cache,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_v2_MulPro,
        pin_memory=torch.cuda.is_available(),
    )

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
    model = load_checkpoint(model, args.model_path, device)

    keys, y_true, y_pred, probabilities, logits = run_prediction(model, dataloader, device)
    prediction_frame = export_predictions(
        args.output_csv,
        keys,
        y_true,
        y_pred,
        probabilities,
        logits,
    )
    metrics = compute_metrics(y_true, y_pred)

    print("\n[Prediction Summary]")
    print(f"N_total:   {metrics['N_total']}")
    print(f"N_labeled: {metrics['N_labeled']}")
    if metrics["N_labeled"]:
        print(f"ACC:       {metrics['ACC']:.4f}")
        print(f"MCC:       {metrics['MCC']:.4f}")
        print(
            "Confusion: "
            f"TN={metrics['TN']} FP={metrics['FP']} "
            f"FN={metrics['FN']} TP={metrics['TP']}"
        )
    else:
        print("Metrics were skipped because no labels matched.")

    print("\n[Predicted Class Counts]")
    print(prediction_frame["pred_name"].value_counts())

    summary_csv = args.summary_csv or args.output_csv.with_name("prediction_summary.csv")
    ensure_parent(summary_csv)
    summary = {
        **metrics,
        "n_pred_agonist": int((y_pred == 0).sum()),
        "n_pred_antagonist": int((y_pred == 1).sum()),
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"[Output] Saved summary: {summary_csv}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
