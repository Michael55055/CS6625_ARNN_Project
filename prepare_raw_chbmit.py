from pathlib import Path
import argparse
import json
import re
import time

import mne
import numpy as np
import pandas as pd


STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
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

SEGMENT_SECONDS = 4
SAMPLING_RATE = 256
SEGMENT_SAMPLES = 1024
RANDOM_SEED = 1111


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
            return None

        # Use the first duplicate, such as T8-P8-0.
        indices.append(matches[0])

    return indices


def parse_summary(summary_file):
    text = summary_file.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    blocks = re.split(
        r"(?=File Name:)",
        text
    )

    seizure_intervals = {}

    for block in blocks:
        file_match = re.search(
            r"File Name:\s*(\S+\.edf)",
            block,
            flags=re.IGNORECASE
        )

        if not file_match:
            continue

        edf_name = file_match.group(1).strip()

        start_times = re.findall(
            r"Seizure(?:\s+\d+)?\s+Start Time:"
            r"\s*(\d+)\s*seconds",
            block,
            flags=re.IGNORECASE
        )

        end_times = re.findall(
            r"Seizure(?:\s+\d+)?\s+End Time:"
            r"\s*(\d+)\s*seconds",
            block,
            flags=re.IGNORECASE
        )

        intervals = []

        for start, end in zip(
            start_times,
            end_times
        ):
            intervals.append(
                (float(start), float(end))
            )

        seizure_intervals[edf_name] = intervals

    return seizure_intervals


def segment_has_seizure(
    segment_start,
    segment_end,
    seizure_intervals
):
    for seizure_start, seizure_end in seizure_intervals:
        if (
            segment_start < seizure_end
            and segment_end > seizure_start
        ):
            return 1

    return 0


def build_manifest(raw_folder, cases):
    rows = []
    excluded_files = []
    valid_files = 0

    for case_number, case_name in enumerate(
        cases,
        start=1
    ):
        case_folder = raw_folder / case_name
        summary_file = (
            case_folder / f"{case_name}-summary.txt"
        )

        if not case_folder.is_dir():
            raise FileNotFoundError(case_folder)

        if not summary_file.is_file():
            raise FileNotFoundError(summary_file)

        seizure_information = parse_summary(
            summary_file
        )

        edf_files = sorted(
            case_folder.glob("*.edf")
        )

        print(
            f"Processing {case_number}/{len(cases)}: "
            f"{case_name} ({len(edf_files)} EDF files)"
        )

        for edf_path in edf_files:
            raw = mne.io.read_raw_edf(
                edf_path,
                preload=False,
                verbose="ERROR"
            )

            sampling_rate = float(
                raw.info["sfreq"]
            )

            channel_indices = get_channel_indices(
                raw
            )

            if (
                sampling_rate != SAMPLING_RATE
                or channel_indices is None
            ):
                excluded_files.append(
                    {
                        "Case": case_name,
                        "EDF_File": edf_path.name,
                        "Sampling_Rate_Hz": sampling_rate,
                        "Reason": (
                            "Missing standard channels "
                            "or unexpected sampling rate"
                        ),
                    }
                )

                raw.close()
                continue

            number_of_segments = (
                int(raw.n_times) // SEGMENT_SAMPLES
            )

            available_channels = len(
                raw.ch_names
            )

            raw.close()
            valid_files += 1

            intervals = seizure_information.get(
                edf_path.name,
                []
            )

            split_name = (
                "train"
                if case_name in TRAIN_CASES
                else "test"
            )

            for segment_index in range(
                number_of_segments
            ):
                start_second = (
                    segment_index * SEGMENT_SECONDS
                )

                end_second = (
                    start_second + SEGMENT_SECONDS
                )

                label = segment_has_seizure(
                    start_second,
                    end_second,
                    intervals
                )

                rows.append(
                    {
                        "Case": case_name,
                        "EDF_File": edf_path.name,
                        "Segment_Index": segment_index,
                        "Start_Second": float(
                            start_second
                        ),
                        "End_Second": float(
                            end_second
                        ),
                        "Label": label,
                        "Class_Name": (
                            "Seizure"
                            if label == 1
                            else "Non-Seizure"
                        ),
                        "Split": split_name,
                        "Sampling_Rate_Hz": (
                            sampling_rate
                        ),
                        "Available_Channels": (
                            available_channels
                        ),
                        "Segment_Samples": (
                            SEGMENT_SAMPLES
                        ),
                    }
                )

    manifest = pd.DataFrame(rows)
    excluded = pd.DataFrame(excluded_files)

    return manifest, excluded, valid_files


def create_balanced_training_manifest(
    manifest,
    seed
):
    training = manifest[
        manifest["Split"].eq("train")
    ].copy()

    seizure = training[
        training["Label"].eq(1)
    ].copy()

    nonseizure = training[
        training["Label"].eq(0)
    ].copy()

    selected_nonseizure = nonseizure.sample(
        n=len(seizure),
        replace=False,
        random_state=seed
    )

    balanced = pd.concat(
        [selected_nonseizure, seizure],
        ignore_index=True
    )

    balanced = balanced.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    balanced.insert(
        0,
        "Cache_Index",
        np.arange(
            len(balanced),
            dtype=np.int64
        )
    )

    return balanced


def create_training_cache(
    balanced,
    raw_folder,
    output_folder
):
    number_of_segments = len(balanced)

    signal_file = (
        output_folder / "raw_training_signals.npy"
    )

    label_file = (
        output_folder / "raw_training_labels.npy"
    )

    signals = np.lib.format.open_memmap(
        signal_file,
        mode="w+",
        dtype=np.float32,
        shape=(
            number_of_segments,
            SEGMENT_SAMPLES,
            len(STANDARD_CHANNELS)
        )
    )

    labels = balanced[
        "Label"
    ].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    np.save(label_file, labels)

    channel_sum = np.zeros(
        len(STANDARD_CHANNELS),
        dtype=np.float64
    )

    channel_squared_sum = np.zeros(
        len(STANDARD_CHANNELS),
        dtype=np.float64
    )

    total_time_samples = 0
    processed_segments = 0

    grouped = balanced.groupby(
        ["Case", "EDF_File"],
        sort=True
    )

    total_files = len(grouped)

    for file_number, (
        (case_name, edf_name),
        rows
    ) in enumerate(grouped, start=1):
        edf_path = (
            raw_folder / case_name / edf_name
        )

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose="ERROR"
        )

        sampling_rate = float(
            raw.info["sfreq"]
        )

        channel_indices = get_channel_indices(
            raw
        )

        if channel_indices is None:
            raw.close()

            raise ValueError(
                f"Missing channels in {edf_path}"
            )

        for row in rows.itertuples(
            index=False
        ):
            start_sample = int(
                round(
                    float(row.Start_Second)
                    * sampling_rate
                )
            )

            stop_sample = (
                start_sample + SEGMENT_SAMPLES
            )

            segment = raw.get_data(
                picks=channel_indices,
                start=start_sample,
                stop=stop_sample
            )

            segment = (
                segment.T.astype(np.float32)
                * 1_000_000.0
            )

            if segment.shape != (
                SEGMENT_SAMPLES,
                len(STANDARD_CHANNELS)
            ):
                raw.close()

                raise ValueError(
                    f"Wrong segment shape: "
                    f"{segment.shape}"
                )

            signals[
                int(row.Cache_Index)
            ] = segment

            channel_sum += segment.sum(
                axis=0,
                dtype=np.float64
            )

            channel_squared_sum += (
                np.square(
                    segment,
                    dtype=np.float64
                ).sum(axis=0)
            )

            total_time_samples += (
                SEGMENT_SAMPLES
            )

            processed_segments += 1

        raw.close()

        if (
            file_number % 10 == 0
            or file_number == total_files
        ):
            print(
                f"Cache progress: "
                f"{file_number}/{total_files} EDF files, "
                f"{processed_segments}/"
                f"{number_of_segments} segments"
            )

    signals.flush()

    channel_mean = (
        channel_sum / total_time_samples
    )

    channel_variance = (
        channel_squared_sum
        / total_time_samples
        - np.square(channel_mean)
    )

    channel_variance = np.maximum(
        channel_variance,
        1e-12
    )

    channel_std = np.sqrt(
        channel_variance
    )

    for start_index in range(
        0,
        number_of_segments,
        50
    ):
        end_index = min(
            start_index + 50,
            number_of_segments
        )

        batch = np.asarray(
            signals[start_index:end_index],
            dtype=np.float32
        )

        signals[start_index:end_index] = (
            batch
            - channel_mean.astype(np.float32)
        ) / channel_std.astype(np.float32)

    signals.flush()
    del signals

    normalization = {
        "random_seed": RANDOM_SEED,
        "sampling_rate_hz": SAMPLING_RATE,
        "segment_seconds": SEGMENT_SECONDS,
        "stride_seconds": SEGMENT_SECONDS,
        "segment_samples": SEGMENT_SAMPLES,
        "selected_channels": STANDARD_CHANNELS,
        "signal_unit_before_normalization": (
            "microvolts"
        ),
        "normalization_source": (
            "Balanced training segments from "
            "chb01 through chb09 only"
        ),
        "channel_mean_microvolts": (
            channel_mean.tolist()
        ),
        "channel_std_microvolts": (
            channel_std.tolist()
        ),
    }

    with open(
        output_folder
        / "training_normalization.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            normalization,
            file,
            indent=2
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare raw CHB-MIT EDF data for "
            "the original ARNN model."
        )
    )

    parser.add_argument(
        "--raw_folder",
        required=True,
        help=(
            "Folder containing chb01 through "
            "chb12"
        )
    )

    parser.add_argument(
        "--output_folder",
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    raw_folder = Path(args.raw_folder)
    output_folder = Path(
        args.output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    cases = TRAIN_CASES + TEST_CASES

    print("=" * 70)
    print("RAW CHB-MIT PREPARATION")
    print("=" * 70)
    print("Cases:", cases)
    print("Training:", TRAIN_CASES)
    print("Testing:", TEST_CASES)
    print("Seed:", args.seed)

    start_time = time.time()

    manifest, excluded, valid_files = (
        build_manifest(
            raw_folder,
            cases
        )
    )

    manifest_file = (
        output_folder
        / "final_clean_segment_manifest.csv"
    )

    excluded_file = (
        output_folder
        / "excluded_edf_files.csv"
    )

    manifest.to_csv(
        manifest_file,
        index=False
    )

    excluded.to_csv(
        excluded_file,
        index=False
    )

    balanced = (
        create_balanced_training_manifest(
            manifest,
            args.seed
        )
    )

    balanced.to_csv(
        output_folder
        / "balanced_training_manifest.csv",
        index=False
    )

    test_manifest = manifest[
        manifest["Split"].eq("test")
    ].copy()

    test_manifest.to_csv(
        output_folder
        / "unchanged_test_manifest.csv",
        index=False
    )

    counts = (
        manifest.groupby(
            ["Split", "Label"]
        )
        .size()
        .reset_index(name="Count")
    )

    counts.to_csv(
        output_folder
        / "class_counts.csv",
        index=False
    )

    create_training_cache(
        balanced,
        raw_folder,
        output_folder
    )

    elapsed_minutes = (
        time.time() - start_time
    ) / 60

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETED")
    print("=" * 70)
    print("Valid EDF files:", valid_files)
    print("Excluded EDF files:", len(excluded))
    print("Manifest rows:", len(manifest))
    print("Balanced training rows:", len(balanced))
    print("Unchanged test rows:", len(test_manifest))
    print(
        f"Elapsed time: {elapsed_minutes:.2f} minutes"
    )
    print("Output:", output_folder)


if __name__ == "__main__":
    main()
