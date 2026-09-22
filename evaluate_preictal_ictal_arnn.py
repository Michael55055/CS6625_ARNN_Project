from pathlib import Path
import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from torch.utils.data import DataLoader, TensorDataset


SCRIPT_FOLDER = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(SCRIPT_FOLDER / "CHB_MIT"),
)

from model import Attentive_RNN


def build_model(device):
    model = Attentive_RNN(
        d_dim=18,
        embed_dim=40,
        seq_len=1024,
        dim_head=10,
        heads=4,
        num_state_vectors=64,
        time_steps=16,
        num_class=1,
        qk_rmsnorm=True,
        rotary_pos_emb=True,
    )

    return model.to(device)


def calculate_metrics(
    true_labels,
    probabilities,
    threshold,
):
    predicted_labels = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    metrics = {
        "Test_Samples": int(len(true_labels)),
        "Preictal_Test_Count": int(
            (true_labels == 0).sum()
        ),
        "Ictal_Test_Count": int(
            (true_labels == 1).sum()
        ),
        "Threshold": float(threshold),
        "Accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "Accuracy_Percent": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            ) * 100
        ),
        "F1": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "PR_AUC": float(
            average_precision_score(
                true_labels,
                probabilities,
            )
        ),
        "ROC_AUC": float(
            roc_auc_score(
                true_labels,
                probabilities,
            )
        ),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "True_Positive": int(tp),
    }

    return metrics, predicted_labels, matrix


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the original ARNN on the raw "
            "CHB-MIT preictal-versus-ictal test cache."
        )
    )

    parser.add_argument(
        "--cache_folder",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output_folder",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    cache_folder = Path(args.cache_folder)
    checkpoint_file = Path(args.checkpoint)
    output_folder = Path(args.output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    signals_file = (
        cache_folder
        / "preictal_ictal_testing_signals.npy"
    )

    labels_file = (
        cache_folder
        / "preictal_ictal_testing_labels.npy"
    )

    manifest_file = (
        cache_folder
        / "preictal_ictal_testing_manifest.csv"
    )

    required_files = [
        signals_file,
        labels_file,
        manifest_file,
        checkpoint_file,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "RAW CHB-MIT PREICTAL-VERSUS-ICTAL "
        "ARNN EVALUATION"
    )
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Threshold:", args.threshold)
    print("Threshold selected on test data: False")

    signals = np.load(
        signals_file
    ).astype(np.float32)

    labels = (
        np.load(labels_file)
        .reshape(-1)
        .astype(np.int64)
    )

    manifest = pd.read_csv(
        manifest_file
    ).reset_index(drop=True)

    print("Signal shape:", signals.shape)
    print("Label shape:", labels.shape)
    print("Manifest rows:", len(manifest))

    if signals.shape != (1048, 1024, 18):
        raise ValueError(
            f"Unexpected signal shape: {signals.shape}"
        )

    if labels.shape != (1048,):
        raise ValueError(
            f"Unexpected label shape: {labels.shape}"
        )

    if len(manifest) != 1048:
        raise ValueError(
            f"Unexpected manifest rows: {len(manifest)}"
        )

    test_cases = sorted(
        manifest["Case"]
        .astype(str)
        .unique()
        .tolist()
    )

    if test_cases != [
        "chb10",
        "chb11",
        "chb12",
    ]:
        raise ValueError(
            f"Unexpected test cases: {test_cases}"
        )

    if int((labels == 0).sum()) != 524:
        raise ValueError(
            "Expected 524 preictal segments."
        )

    if int((labels == 1).sum()) != 524:
        raise ValueError(
            "Expected 524 ictal segments."
        )

    if not np.isfinite(signals).all():
        raise ValueError(
            "Test signals contain nonfinite values."
        )

    model = build_model(device)

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(signals),
        torch.from_numpy(labels),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    probabilities = []
    true_labels = []

    evaluation_start = time.time()

    with torch.no_grad():
        for batch_number, (
            batch_signals,
            batch_labels,
        ) in enumerate(loader, start=1):
            batch_signals = batch_signals.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            batch_probabilities = (
                model(batch_signals)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            probabilities.extend(
                batch_probabilities.tolist()
            )

            true_labels.extend(
                batch_labels.numpy().tolist()
            )

            if (
                batch_number % 5 == 0
                or batch_number == len(loader)
            ):
                completed = min(
                    batch_number * args.batch_size,
                    len(labels),
                )

                print(
                    f"Batch {batch_number}/{len(loader)} "
                    f"| Segments {completed}/{len(labels)}"
                )

    evaluation_seconds = (
        time.time() - evaluation_start
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    true_labels = np.asarray(
        true_labels,
        dtype=int,
    )

    metrics, predicted_labels, matrix = (
        calculate_metrics(
            true_labels,
            probabilities,
            args.threshold,
        )
    )

    metrics.update(
        {
            "Dataset": (
                "CHB-MIT Raw EDF "
                "Preictal vs Ictal"
            ),
            "Negative_Class": "Preictal",
            "Positive_Class": "Ictal",
            "Preictal_Definition": (
                "Immediately before each seizure, "
                "with duration equal to that seizure"
            ),
            "Training_Cases": "chb01-chb09",
            "Testing_Cases": "chb10-chb12",
            "Patient_Wise_Split": True,
            "Patient_Overlap": False,
            "Threshold_Selected_On_Test": False,
            "Epochs": int(
                checkpoint.get("epochs", 30)
            ),
            "Batch_Size": int(
                checkpoint.get(
                    "batch_size",
                    args.batch_size,
                )
            ),
            "Training_Seconds": float(
                checkpoint.get(
                    "total_training_seconds",
                    np.nan,
                )
            ),
            "Evaluation_Seconds": float(
                evaluation_seconds
            ),
            "Evaluation_Device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),
        }
    )

    predictions = manifest.copy()
    predictions["True_Label"] = true_labels
    predictions["Predicted_Label"] = (
        predicted_labels
    )
    predictions["Ictal_Probability"] = (
        probabilities
    )

    predictions.to_csv(
        output_folder
        / "preictal_ictal_test_predictions.csv",
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        output_folder
        / "preictal_ictal_metrics.csv",
        index=False,
    )

    confusion_frame = pd.DataFrame(
        matrix,
        index=[
            "Actual_Preictal",
            "Actual_Ictal",
        ],
        columns=[
            "Predicted_Preictal",
            "Predicted_Ictal",
        ],
    )

    confusion_frame.to_csv(
        output_folder
        / "preictal_ictal_confusion_matrix.csv"
    )

    patient_rows = []

    for case_name, case_frame in (
        predictions.groupby("Case", sort=True)
    ):
        case_true = case_frame[
            "True_Label"
        ].to_numpy(dtype=int)

        case_probability = case_frame[
            "Ictal_Probability"
        ].to_numpy(dtype=float)

        case_metrics, _, _ = calculate_metrics(
            case_true,
            case_probability,
            args.threshold,
        )

        case_metrics["Case"] = case_name
        patient_rows.append(case_metrics)

    pd.DataFrame(patient_rows).to_csv(
        output_folder
        / "preictal_ictal_metrics_by_patient.csv",
        index=False,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Preictal",
            "Ictal",
        ],
    )

    display.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title(
        "Raw CHB-MIT Preictal-Ictal "
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_folder
        / "preictal_ictal_confusion_matrix.png",
        dpi=200,
    )

    plt.close()

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            true_labels,
            probabilities,
        )
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        recall_curve,
        precision_curve,
        label=(
            f"PR-AUC = "
            f"{metrics['PR_AUC']:.4f}"
        ),
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Raw CHB-MIT Preictal-Ictal "
        "Precision-Recall Curve"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_folder
        / "preictal_ictal_pr_curve.png",
        dpi=200,
    )

    plt.close()

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            true_labels,
            probabilities,
        )
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=(
            f"ROC-AUC = "
            f"{metrics['ROC_AUC']:.4f}"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "Raw CHB-MIT Preictal-Ictal "
        "ROC Curve"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_folder
        / "preictal_ictal_roc_curve.png",
        dpi=200,
    )

    plt.close()

    configuration = {
        "threshold": float(args.threshold),
        "threshold_selected_on_test": False,
        "training_cases": [
            f"chb{i:02d}"
            for i in range(1, 10)
        ],
        "testing_cases": [
            "chb10",
            "chb11",
            "chb12",
        ],
        "negative_class": "Preictal",
        "positive_class": "Ictal",
        "preictal_definition": (
            "Immediately before each seizure, "
            "with duration equal to that seizure"
        ),
    }

    with open(
        output_folder
        / "evaluation_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )

    log_lines = [
        (
            "RAW CHB-MIT PREICTAL-VERSUS-ICTAL "
            "ARNN EVALUATION"
        ),
        "",
        f"Device: {metrics['Evaluation_Device']}",
        "Training patients: chb01-chb09",
        "Testing patients: chb10-chb12",
        "Patient overlap: none",
        f"Threshold: {args.threshold}",
        "Threshold selected on test data: no",
        "",
        f"Test samples: {len(true_labels)}",
        (
            "Preictal samples: "
            f"{int((true_labels == 0).sum())}"
        ),
        (
            "Ictal samples: "
            f"{int((true_labels == 1).sum())}"
        ),
        "",
        f"Accuracy: {metrics['Accuracy']:.6f}",
        f"F1: {metrics['F1']:.6f}",
        f"PR-AUC: {metrics['PR_AUC']:.6f}",
        f"ROC-AUC: {metrics['ROC_AUC']:.6f}",
        (
            "Sensitivity: "
            f"{metrics['Sensitivity']:.6f}"
        ),
        (
            "Specificity: "
            f"{metrics['Specificity']:.6f}"
        ),
        "",
        (
            "Evaluation seconds: "
            f"{evaluation_seconds:.2f}"
        ),
    ]

    (
        output_folder
        / "evaluation_console_log.txt"
    ).write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(confusion_frame)

    print("\nOutput folder:")
    print(output_folder)


if __name__ == "__main__":
    main()
