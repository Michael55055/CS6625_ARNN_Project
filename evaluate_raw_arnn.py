from pathlib import Path
import argparse
import json
import re
import sys
import time

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

SCRIPT_FOLDER = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(SCRIPT_FOLDER / "CHB_MIT")
)

from model import Attentive_RNN


STANDARD_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]


def write_log(message, log_file):
    print(message, flush=True)

    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(str(message) + "\n")


def normalize_channel_name(name):
    name = str(name).strip().upper()
    name = name.replace("EEG ", "")
    name = name.replace(" ", "")
    name = re.sub(r"-\d+$", "", name)
    return name


def get_channel_indices(raw):
    normalized_names = [
        normalize_channel_name(name)
        for name in raw.ch_names
    ]

    indices = []

    for required_channel in STANDARD_CHANNELS:
        required = normalize_channel_name(
            required_channel
        )

        matches = [
            index
            for index, current_name
            in enumerate(normalized_names)
            if current_name == required
        ]

        if not matches:
            raise ValueError(
                f"Missing required channel: "
                f"{required_channel}"
            )

        # Use T8-P8-0 instead of the duplicate T8-P8-1.
        indices.append(matches[0])

    return indices


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
        rotary_pos_emb=True
    )

    return model.to(device)


def calculate_metrics(
    true_labels,
    probabilities,
    threshold=0.5
):
    predicted_labels = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "Test_Samples": int(len(true_labels)),
        "Nonseizure_Test_Count": int(
            (true_labels == 0).sum()
        ),
        "Seizure_Test_Count": int(
            (true_labels == 1).sum()
        ),
        "Threshold": float(threshold),
        "Accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels
            )
        ),
        "Accuracy_Percent": float(
            accuracy_score(
                true_labels,
                predicted_labels
            ) * 100
        ),
        "F1": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0
            )
        ),
        "PR_AUC": float(
            average_precision_score(
                true_labels,
                probabilities
            )
        ),
        "ROC_AUC": float(
            roc_auc_score(
                true_labels,
                probabilities
            )
        ),
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "True_Positive": int(tp),
    }

    return metrics, predicted_labels, matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_folder",
        required=True
    )

    parser.add_argument(
        "--raw_edf_folder",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--output_folder",
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    cache_folder = Path(args.cache_folder)
    raw_edf_folder = Path(args.raw_edf_folder)
    checkpoint_file = Path(args.checkpoint)
    output_folder = Path(args.output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = (
        output_folder / "evaluation_console_log.txt"
    )

    if log_file.exists():
        log_file.unlink()

    test_manifest_file = (
        cache_folder / "unchanged_test_manifest.csv"
    )

    normalization_file = (
        cache_folder / "training_normalization.json"
    )

    if not test_manifest_file.is_file():
        raise FileNotFoundError(
            test_manifest_file
        )

    if not normalization_file.is_file():
        raise FileNotFoundError(
            normalization_file
        )

    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            checkpoint_file
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    write_log("=" * 70, log_file)
    write_log(
        "RAW CHB-MIT PATIENT-WISE ARNN EVALUATION",
        log_file
    )
    write_log("=" * 70, log_file)

    write_log(
        f"Device: {device}",
        log_file
    )

    if torch.cuda.is_available():
        write_log(
            f"GPU: {torch.cuda.get_device_name(0)}",
            log_file
        )

    write_log(
        f"Batch size: {args.batch_size}",
        log_file
    )

    write_log(
        f"Threshold: {args.threshold}",
        log_file
    )

    test_manifest = pd.read_csv(
        test_manifest_file
    )

    test_manifest = test_manifest[
        test_manifest["Split"]
        .str.lower()
        .eq("test")
    ].copy()

    test_manifest = test_manifest.sort_values(
        [
            "Case",
            "EDF_File",
            "Segment_Index"
        ]
    ).reset_index(drop=True)

    test_cases = sorted(
        test_manifest["Case"].unique()
    )

    if test_cases != [
        "chb10",
        "chb11",
        "chb12"
    ]:
        raise ValueError(
            f"Unexpected test cases: {test_cases}"
        )

    if len(test_manifest) != 94950:
        raise ValueError(
            f"Unexpected test count: "
            f"{len(test_manifest)}"
        )

    write_log(
        f"Test patients: {test_cases}",
        log_file
    )

    write_log(
        f"Test segments: {len(test_manifest)}",
        log_file
    )

    write_log(
        "Nonseizure test segments: "
        f"{int((test_manifest['Label'] == 0).sum())}",
        log_file
    )

    write_log(
        "Seizure test segments: "
        f"{int((test_manifest['Label'] == 1).sum())}",
        log_file
    )

    with open(
        normalization_file,
        "r",
        encoding="utf-8"
    ) as file:
        normalization = json.load(file)

    channel_mean = np.asarray(
        normalization["channel_mean_microvolts"],
        dtype=np.float32
    )

    channel_std = np.asarray(
        normalization["channel_std_microvolts"],
        dtype=np.float32
    )

    if channel_mean.shape != (18,):
        raise ValueError(
            f"Wrong mean shape: {channel_mean.shape}"
        )

    if channel_std.shape != (18,):
        raise ValueError(
            f"Wrong std shape: {channel_std.shape}"
        )

    model = build_model(device)

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_file,
            map_location=device
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    write_log(
        "Checkpoint loaded successfully.",
        log_file
    )

    predictions = []
    evaluation_start = time.time()

    grouped = test_manifest.groupby(
        ["Case", "EDF_File"],
        sort=True
    )

    total_edf_files = len(grouped)
    processed_edf_files = 0
    processed_segments = 0

    with torch.no_grad():
        for (case_name, edf_name), rows in grouped:
            edf_path = (
                raw_edf_folder
                / case_name
                / edf_name
            )

            if not edf_path.is_file():
                raise FileNotFoundError(
                    edf_path
                )

            raw = mne.io.read_raw_edf(
                edf_path,
                preload=False,
                verbose="ERROR"
            )

            sampling_rate = float(
                raw.info["sfreq"]
            )

            if sampling_rate != 256.0:
                raw.close()

                raise ValueError(
                    f"Unexpected sampling rate in "
                    f"{edf_name}: {sampling_rate}"
                )

            channel_indices = (
                get_channel_indices(raw)
            )

            # Load the selected 18 channels once per EDF.
            edf_signal = raw.get_data(
                picks=channel_indices
            )

            raw.close()

            edf_signal = (
                edf_signal.T.astype(np.float32)
                * 1_000_000.0
            )

            rows = rows.sort_values(
                "Segment_Index"
            ).reset_index(drop=True)

            for batch_start in range(
                0,
                len(rows),
                args.batch_size
            ):
                batch_rows = rows.iloc[
                    batch_start:
                    batch_start + args.batch_size
                ]

                segment_list = []

                for row in batch_rows.itertuples(
                    index=False
                ):
                    start_sample = int(
                        round(
                            float(row.Start_Second)
                            * sampling_rate
                        )
                    )

                    stop_sample = (
                        start_sample + 1024
                    )

                    segment = edf_signal[
                        start_sample:stop_sample
                    ]

                    if segment.shape != (1024, 18):
                        raise ValueError(
                            f"Wrong segment shape in "
                            f"{case_name}/{edf_name}: "
                            f"{segment.shape}"
                        )

                    segment_list.append(segment)

                batch = np.stack(
                    segment_list,
                    axis=0
                ).astype(np.float32)

                batch = (
                    batch - channel_mean
                ) / channel_std

                data = torch.from_numpy(batch).to(
                    device,
                    non_blocking=True
                )

                probability = model(data)

                probability = (
                    probability.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                for row, probability_value in zip(
                    batch_rows.itertuples(
                        index=False
                    ),
                    probability
                ):
                    predictions.append(
                        {
                            "Case": row.Case,
                            "EDF_File": row.EDF_File,
                            "Segment_Index": int(
                                row.Segment_Index
                            ),
                            "Start_Second": float(
                                row.Start_Second
                            ),
                            "End_Second": float(
                                row.End_Second
                            ),
                            "True_Label": int(
                                row.Label
                            ),
                            "Seizure_Probability": float(
                                probability_value
                            ),
                        }
                    )

                processed_segments += len(
                    batch_rows
                )

            processed_edf_files += 1

            elapsed_minutes = (
                time.time() - evaluation_start
            ) / 60

            write_log(
                (
                    f"Processed EDF "
                    f"{processed_edf_files}/"
                    f"{total_edf_files} | "
                    f"Segments "
                    f"{processed_segments}/"
                    f"{len(test_manifest)} | "
                    f"{elapsed_minutes:.2f} minutes"
                ),
                log_file
            )

            # Save resumable progress after each EDF.
            pd.DataFrame(predictions).to_csv(
                output_folder
                / "evaluation_progress.csv",
                index=False
            )

    evaluation_seconds = (
        time.time() - evaluation_start
    )

    predictions_frame = pd.DataFrame(
        predictions
    )

    if len(predictions_frame) != 94950:
        raise RuntimeError(
            "Prediction count is incorrect: "
            f"{len(predictions_frame)}"
        )

    true_labels = predictions_frame[
        "True_Label"
    ].to_numpy(dtype=int)

    probabilities = predictions_frame[
        "Seizure_Probability"
    ].to_numpy(dtype=float)

    metrics, predicted_labels, matrix = (
        calculate_metrics(
            true_labels,
            probabilities,
            threshold=args.threshold
        )
    )

    predictions_frame[
        "Predicted_Label"
    ] = predicted_labels

    metrics.update(
        {
            "Dataset": "CHB-MIT Raw EDF",
            "Training_Cases": (
                "chb01-chb09"
            ),
            "Testing_Cases": (
                "chb10-chb12"
            ),
            "Patient_Wise_Split": True,
            "Patient_Overlap": False,
            "Epochs": int(
                checkpoint["epochs"]
            ),
            "Batch_Size": int(
                checkpoint["batch_size"]
            ),
            "Training_Seconds": float(
                checkpoint[
                    "total_training_seconds"
                ]
            ),
            "Evaluation_Seconds": float(
                evaluation_seconds
            ),
        }
    )

    predictions_file = (
        output_folder
        / "raw_chb_mit_test_predictions.csv"
    )

    metrics_file = (
        output_folder
        / "raw_chb_mit_metrics.csv"
    )

    confusion_file = (
        output_folder
        / "raw_chb_mit_confusion_matrix.csv"
    )

    predictions_frame.to_csv(
        predictions_file,
        index=False
    )

    pd.DataFrame([metrics]).to_csv(
        metrics_file,
        index=False
    )

    confusion_frame = pd.DataFrame(
        matrix,
        index=[
            "Actual_Nonseizure",
            "Actual_Seizure"
        ],
        columns=[
            "Predicted_Nonseizure",
            "Predicted_Seizure"
        ]
    )

    confusion_frame.to_csv(
        confusion_file
    )

    # Per-patient results
    patient_results = []

    for case_name, case_frame in (
        predictions_frame.groupby("Case")
    ):
        case_true = case_frame[
            "True_Label"
        ].to_numpy(dtype=int)

        case_probability = case_frame[
            "Seizure_Probability"
        ].to_numpy(dtype=float)

        case_metrics, _, case_matrix = (
            calculate_metrics(
                case_true,
                case_probability,
                threshold=args.threshold
            )
        )

        case_metrics["Case"] = case_name
        patient_results.append(case_metrics)

    pd.DataFrame(patient_results).to_csv(
        output_folder
        / "raw_chb_mit_metrics_by_patient.csv",
        index=False
    )

    # Confusion matrix figure
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Nonseizure",
            "Seizure"
        ]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Raw CHB-MIT ARNN Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_folder
        / "raw_chb_mit_confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # Precision-recall curve
    precision, recall, _ = (
        precision_recall_curve(
            true_labels,
            probabilities
        )
    )

    plt.figure(figsize=(6, 5))
    plt.plot(
        recall,
        precision,
        label=(
            f"PR-AUC = "
            f"{metrics['PR_AUC']:.4f}"
        )
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Raw CHB-MIT Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_folder
        / "raw_chb_mit_pr_curve.png",
        dpi=200
    )

    plt.close()

    # ROC curve
    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            true_labels,
            probabilities
        )
    )

    plt.figure(figsize=(6, 5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=(
            f"ROC-AUC = "
            f"{metrics['ROC_AUC']:.4f}"
        )
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Raw CHB-MIT ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_folder
        / "raw_chb_mit_roc_curve.png",
        dpi=200
    )

    plt.close()

    write_log("=" * 70, log_file)
    write_log(
        "FINAL RAW CHB-MIT RESULTS",
        log_file
    )
    write_log("=" * 70, log_file)

    for metric_name, metric_value in metrics.items():
        write_log(
            f"{metric_name}: {metric_value}",
            log_file
        )

    write_log(
        "\nConfusion matrix:",
        log_file
    )

    write_log(
        str(confusion_frame),
        log_file
    )

    write_log(
        f"\nPredictions saved: "
        f"{predictions_file}",
        log_file
    )

    write_log(
        f"Metrics saved: {metrics_file}",
        log_file
    )

    write_log(
        "EVALUATION COMPLETED SUCCESSFULLY.",
        log_file
    )


if __name__ == "__main__":
    main()
