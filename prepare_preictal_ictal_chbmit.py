from pathlib import Path
import argparse
import json
import re
import time

import mne
import numpy as np
import pandas as pd


CHANNELS = [
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

TRAIN_CASES = [
    f"chb{number:02d}"
    for number in range(1, 10)
]

TEST_CASES = [
    "chb10",
    "chb11",
    "chb12",
]

SAMPLING_RATE = 256
SEGMENT_SECONDS = 4
SEGMENT_SAMPLES = 1024


def normalize_channel_name(name):
    name = str(name).strip().upper()
    name = name.replace("EEG ", "")
    name = name.replace(" ", "")
    name = re.sub(r"-\d+$", "", name)
    return name


def select_channel_indices(channel_names):
    normalized_names = [
        normalize_channel_name(name)
        for name in channel_names
    ]

    selected_indices = []

    for required_channel in CHANNELS:
        required_name = normalize_channel_name(
            required_channel
        )

        matches = [
            index
            for index, current_name
            in enumerate(normalized_names)
            if current_name == required_name
        ]

        if not matches:
            raise ValueError(
                f"Missing channel {required_channel}. "
                f"Available channels: {channel_names}"
            )

        selected_indices.append(matches[0])

    return selected_indices


def validate_manifest(manifest):
    required_columns = [
        "Event_ID",
        "Case",
        "Split",
        "Class_Name",
        "Label",
        "EDF_File",
        "Segment_Index_In_Event",
        "Start_Second_In_EDF",
        "End_Second_In_EDF",
        "Segment_Seconds",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Manifest columns are missing: "
            f"{missing_columns}"
        )

    if len(manifest) != 2830:
        raise ValueError(
            f"Expected 2,830 manifest rows, "
            f"found {len(manifest)}."
        )

    training_cases = sorted(
        manifest.loc[
            manifest["Split"].eq("train"),
            "Case",
        ].astype(str).unique().tolist()
    )

    testing_cases = sorted(
        manifest.loc[
            manifest["Split"].eq("test"),
            "Case",
        ].astype(str).unique().tolist()
    )

    if training_cases != TRAIN_CASES:
        raise ValueError(
            f"Unexpected training cases: "
            f"{training_cases}"
        )

    if testing_cases != TEST_CASES:
        raise ValueError(
            f"Unexpected testing cases: "
            f"{testing_cases}"
        )

    patient_overlap = set(
        training_cases
    ).intersection(testing_cases)

    if patient_overlap:
        raise ValueError(
            f"Patient leakage detected: "
            f"{sorted(patient_overlap)}"
        )

    expected_counts = {
        ("train", 0): 891,
        ("train", 1): 891,
        ("test", 0): 524,
        ("test", 1): 524,
    }

    actual_counts = (
        manifest.groupby(
            ["Split", "Label"]
        ).size().to_dict()
    )

    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected class counts: "
            f"{actual_counts}"
        )

    if not np.allclose(
        manifest["Segment_Seconds"].to_numpy(
            dtype=float
        ),
        SEGMENT_SECONDS,
    ):
        raise ValueError(
            "The manifest contains a segment "
            "length other than four seconds."
        )


def extract_signals(
    manifest,
    raw_folder,
):
    manifest = manifest.copy()
    manifest["Cache_Row"] = np.arange(
        len(manifest)
    )

    signals = np.empty(
        (
            len(manifest),
            SEGMENT_SAMPLES,
            len(CHANNELS),
        ),
        dtype=np.float32,
    )

    unique_files = (
        manifest[
            ["Case", "EDF_File"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        "Unique EDF files:",
        len(unique_files),
    )

    extraction_start = time.time()
    completed_segments = 0

    for file_number, file_row in enumerate(
        unique_files.itertuples(index=False),
        start=1,
    ):
        edf_path = (
            raw_folder
            / str(file_row.Case)
            / str(file_row.EDF_File)
        )

        if not edf_path.is_file():
            raise FileNotFoundError(edf_path)

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=False,
            verbose="ERROR",
        )

        sampling_rate = float(
            raw.info["sfreq"]
        )

        if not np.isclose(
            sampling_rate,
            SAMPLING_RATE,
        ):
            raw.close()

            raise ValueError(
                f"Unexpected sampling rate in "
                f"{edf_path}: {sampling_rate}"
            )

        channel_indices = (
            select_channel_indices(
                raw.ch_names
            )
        )

        file_rows = manifest[
            manifest["Case"]
            .astype(str)
            .eq(str(file_row.Case))
            & manifest["EDF_File"]
            .astype(str)
            .eq(str(file_row.EDF_File))
        ]

        for row in file_rows.itertuples(
            index=False
        ):
            start_sample = int(
                round(
                    float(
                        row.Start_Second_In_EDF
                    )
                    * sampling_rate
                )
            )

            stop_sample = (
                start_sample
                + SEGMENT_SAMPLES
            )

            if (
                start_sample < 0
                or stop_sample > raw.n_times
            ):
                raw.close()

                raise ValueError(
                    "Segment exceeds EDF boundaries: "
                    f"{row.Case}/{row.EDF_File}, "
                    f"{row.Start_Second_In_EDF}-"
                    f"{row.End_Second_In_EDF}"
                )

            segment = raw.get_data(
                picks=channel_indices,
                start=start_sample,
                stop=stop_sample,
            )

            if segment.shape != (
                len(CHANNELS),
                SEGMENT_SAMPLES,
            ):
                raw.close()

                raise ValueError(
                    f"Unexpected segment shape in "
                    f"{row.EDF_File}: "
                    f"{segment.shape}"
                )

            segment = (
                segment.T.astype(np.float32)
                * 1_000_000.0
            )

            if not np.isfinite(
                segment
            ).all():
                raw.close()

                raise ValueError(
                    f"Nonfinite values found in "
                    f"{row.EDF_File}."
                )

            signals[
                int(row.Cache_Row)
            ] = segment

            completed_segments += 1

        raw.close()

        if (
            file_number % 5 == 0
            or file_number
            == len(unique_files)
        ):
            elapsed_minutes = (
                time.time()
                - extraction_start
            ) / 60

            print(
                f"Processed {file_number}/"
                f"{len(unique_files)} EDF files "
                f"| {completed_segments}/"
                f"{len(manifest)} segments "
                f"| {elapsed_minutes:.2f} minutes"
            )

    if completed_segments != len(manifest):
        raise RuntimeError(
            f"Expected {len(manifest)} segments, "
            f"extracted {completed_segments}."
        )

    return signals


def normalize_and_save(
    signals,
    manifest,
    output_folder,
):
    training_mask = (
        manifest["Split"]
        .eq("train")
        .to_numpy()
    )

    testing_mask = (
        manifest["Split"]
        .eq("test")
        .to_numpy()
    )

    training_signals = (
        signals[training_mask].copy()
    )

    testing_signals = (
        signals[testing_mask].copy()
    )

    training_labels = (
        manifest.loc[
            training_mask,
            "Label",
        ]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )

    testing_labels = (
        manifest.loc[
            testing_mask,
            "Label",
        ]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )

    training_manifest = (
        manifest.loc[training_mask]
        .reset_index(drop=True)
    )

    testing_manifest = (
        manifest.loc[testing_mask]
        .reset_index(drop=True)
    )

    channel_means = (
        training_signals.mean(
            axis=(0, 1),
            dtype=np.float64,
        )
    )

    channel_standard_deviations = (
        training_signals.std(
            axis=(0, 1),
            dtype=np.float64,
        )
    )

    if np.any(
        channel_standard_deviations <= 0
    ):
        raise ValueError(
            "A training channel has zero "
            "standard deviation."
        )

    means = channel_means.astype(
        np.float32
    )

    standard_deviations = (
        channel_standard_deviations
        .astype(np.float32)
    )

    for beginning in range(
        0,
        len(training_signals),
        128,
    ):
        ending = min(
            beginning + 128,
            len(training_signals),
        )

        training_signals[
            beginning:ending
        ] = (
            training_signals[
                beginning:ending
            ] - means
        ) / standard_deviations

    for beginning in range(
        0,
        len(testing_signals),
        128,
    ):
        ending = min(
            beginning + 128,
            len(testing_signals),
        )

        testing_signals[
            beginning:ending
        ] = (
            testing_signals[
                beginning:ending
            ] - means
        ) / standard_deviations

    if not np.isfinite(
        training_signals
    ).all():
        raise ValueError(
            "Training signals contain "
            "nonfinite values."
        )

    if not np.isfinite(
        testing_signals
    ).all():
        raise ValueError(
            "Testing signals contain "
            "nonfinite values."
        )

    np.save(
        output_folder
        / "preictal_ictal_training_signals.npy",
        training_signals,
    )

    np.save(
        output_folder
        / "preictal_ictal_training_labels.npy",
        training_labels,
    )

    np.save(
        output_folder
        / "preictal_ictal_testing_signals.npy",
        testing_signals,
    )

    np.save(
        output_folder
        / "preictal_ictal_testing_labels.npy",
        testing_labels,
    )

    training_manifest.to_csv(
        output_folder
        / "preictal_ictal_training_manifest.csv",
        index=False,
    )

    testing_manifest.to_csv(
        output_folder
        / "preictal_ictal_testing_manifest.csv",
        index=False,
    )

    normalization = {
        "unit_before_normalization": (
            "microvolts"
        ),
        "statistics_source": (
            "training patients "
            "chb01-chb09 only"
        ),
        "training_cases": TRAIN_CASES,
        "testing_cases": TEST_CASES,
        "sampling_rate_hz": (
            SAMPLING_RATE
        ),
        "segment_seconds": (
            SEGMENT_SECONDS
        ),
        "segment_samples": (
            SEGMENT_SAMPLES
        ),
        "channels": CHANNELS,
        "channel_means": (
            channel_means.tolist()
        ),
        "channel_standard_deviations": (
            channel_standard_deviations
            .tolist()
        ),
    }

    with open(
        output_folder
        / "preictal_ictal_normalization.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            normalization,
            file,
            indent=2,
        )

    summary = [
        "RAW CHB-MIT PREICTAL-ICTAL CACHE",
        "",
        "Preictal definition:",
        (
            "Immediately before each seizure, "
            "with duration equal to that seizure."
        ),
        (
            "Only complete nonoverlapping "
            "four-second segments are retained."
        ),
        "",
        "Training patients: chb01-chb09",
        "Testing patients: chb10-chb12",
        "Patient overlap: none",
        "Sampling rate: 256 Hz",
        "Channels: 18",
        "Segment length: 4 seconds",
        "Stride: 4 seconds",
        "",
        (
            "Training preictal segments: "
            f"{int((training_labels == 0).sum())}"
        ),
        (
            "Training ictal segments: "
            f"{int((training_labels == 1).sum())}"
        ),
        (
            "Testing preictal segments: "
            f"{int((testing_labels == 0).sum())}"
        ),
        (
            "Testing ictal segments: "
            f"{int((testing_labels == 1).sum())}"
        ),
        "",
        (
            "Normalization statistics source: "
            "training patients only"
        ),
    ]

    (
        output_folder
        / "preictal_ictal_cache_summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\nCache verification:")
    print(
        "Training signals:",
        training_signals.shape,
    )
    print(
        "Training labels:",
        training_labels.shape,
    )
    print(
        "Testing signals:",
        testing_signals.shape,
    )
    print(
        "Testing labels:",
        testing_labels.shape,
    )
    print(
        "Normalized training mean:",
        float(training_signals.mean()),
    )
    print(
        "Normalized training standard deviation:",
        float(training_signals.std()),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract model-ready preictal and "
            "ictal segments from raw CHB-MIT EDF "
            "files using an audited manifest."
        )
    )

    parser.add_argument(
        "--raw_folder",
        required=True,
        help=(
            "Folder containing chb01 through "
            "chb12 raw EDF folders."
        ),
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Audited preictal_ictal_manifest.csv."
        ),
    )

    parser.add_argument(
        "--output_folder",
        required=True,
    )

    args = parser.parse_args()

    raw_folder = Path(
        args.raw_folder
    )

    manifest_file = Path(
        args.manifest
    )

    output_folder = Path(
        args.output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not raw_folder.is_dir():
        raise FileNotFoundError(
            raw_folder
        )

    if not manifest_file.is_file():
        raise FileNotFoundError(
            manifest_file
        )

    print("=" * 70)
    print(
        "RAW CHB-MIT PREICTAL-ICTAL "
        "CACHE PREPARATION"
    )
    print("=" * 70)

    print("Raw EDF folder:", raw_folder)
    print("Manifest:", manifest_file)
    print("Output folder:", output_folder)

    print("\nProtocol:")
    print(
        "Preictal = interval immediately "
        "before each seizure, with duration "
        "equal to that seizure."
    )
    print(
        "Training patients: chb01-chb09"
    )
    print(
        "Testing patients: chb10-chb12"
    )
    print(
        "Normalization source: "
        "training patients only"
    )

    manifest = pd.read_csv(
        manifest_file
    ).reset_index(drop=True)

    validate_manifest(manifest)

    print("\nManifest verification passed.")
    print("Manifest rows:", len(manifest))

    signals = extract_signals(
        manifest,
        raw_folder,
    )

    normalize_and_save(
        signals,
        manifest,
        output_folder,
    )

    print("\n" + "=" * 70)
    print("CACHE PREPARATION COMPLETED")
    print("=" * 70)
    print("Output:", output_folder)


if __name__ == "__main__":
    main()
