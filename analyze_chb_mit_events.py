
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate event-level ictal sensitivity for the "
            "CHB-MIT raw-EDF preictal-versus-ictal experiment."
        )
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to preictal_ictal_test_predictions.csv",
    )

    parser.add_argument(
        "--event_audit",
        required=True,
        help="Path to preictal_ictal_event_audit.csv",
    )

    parser.add_argument(
        "--output_folder",
        required=True,
        help="Folder for event-analysis results",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prespecified classification threshold",
    )

    parser.add_argument(
        "--long_event_id",
        default="chb11_event_003",
        help="Event ID used for the post hoc influence check",
    )

    return parser.parse_args()


def verify_columns(data, required_columns, file_description):
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"{file_description} is missing columns: "
            f"{sorted(missing)}"
        )


def main():
    args = parse_arguments()

    prediction_path = Path(args.predictions)
    audit_path = Path(args.event_audit)
    output_folder = Path(args.output_folder)

    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    if not audit_path.exists():
        raise FileNotFoundError(audit_path)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(prediction_path)
    event_audit = pd.read_csv(audit_path)

    prediction_columns = {
        "Event_ID",
        "Case",
        "EDF_File",
        "True_Label",
        "Predicted_Label",
        "Ictal_Probability",
    }

    audit_columns = {
        "Event_ID",
        "Case",
        "Split",
        "Seizure_EDF_File",
        "Seizure_Duration_Seconds",
        "Ictal_Segments",
        "Status",
        "Exclusion_Reason",
    }

    verify_columns(
        predictions,
        prediction_columns,
        "Prediction file",
    )

    verify_columns(
        event_audit,
        audit_columns,
        "Event-audit file",
    )

    if len(predictions) != 1048:
        raise ValueError(
            f"Expected 1048 predictions, found {len(predictions)}"
        )

    expected_predictions = (
        predictions["Ictal_Probability"].astype(float)
        >= args.threshold
    ).astype(int)

    if not np.array_equal(
        expected_predictions.to_numpy(),
        predictions["Predicted_Label"].astype(int).to_numpy(),
    ):
        raise ValueError(
            "Saved predicted labels do not match the "
            f"prespecified threshold of {args.threshold}."
        )

    class_counts = (
        predictions["True_Label"]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    if class_counts.to_dict() != {0: 524, 1: 524}:
        raise ValueError(
            f"Unexpected test class counts: {class_counts.to_dict()}"
        )

    included_test_events = event_audit[
        (event_audit["Split"] == "test")
        & (event_audit["Status"] == "INCLUDED")
    ].copy()

    ictal_predictions = predictions[
        predictions["True_Label"].astype(int) == 1
    ].copy()

    ictal_predictions["Correct_Ictal_Prediction"] = (
        ictal_predictions["Predicted_Label"].astype(int) == 1
    ).astype(int)

    event_results = (
        ictal_predictions
        .groupby(
            [
                "Event_ID",
                "Case",
                "EDF_File",
            ],
            as_index=False,
        )
        .agg(
            Ictal_Segment_Count=(
                "True_Label",
                "size",
            ),
            Correct_Ictal_Segments=(
                "Correct_Ictal_Prediction",
                "sum",
            ),
            Mean_Ictal_Probability=(
                "Ictal_Probability",
                "mean",
            ),
        )
    )

    event_results["Event_Ictal_Sensitivity"] = (
        event_results["Correct_Ictal_Segments"]
        / event_results["Ictal_Segment_Count"]
    )

    audit_columns_for_merge = included_test_events[
        [
            "Event_ID",
            "Seizure_EDF_File",
            "Seizure_Duration_Seconds",
            "Ictal_Segments",
        ]
    ].copy()

    event_results = event_results.merge(
        audit_columns_for_merge,
        on="Event_ID",
        how="left",
        validate="one_to_one",
    )

    if event_results[
        "Seizure_Duration_Seconds"
    ].isna().any():
        raise ValueError(
            "At least one prediction event is missing "
            "from the event audit."
        )

    if not (
        event_results["EDF_File"]
        == event_results["Seizure_EDF_File"]
    ).all():
        raise ValueError(
            "EDF filenames do not match the event audit."
        )

    if not (
        event_results["Ictal_Segment_Count"]
        == event_results["Ictal_Segments"]
    ).all():
        raise ValueError(
            "Ictal segment counts do not match the event audit."
        )

    event_results = event_results[
        [
            "Event_ID",
            "Case",
            "EDF_File",
            "Seizure_Duration_Seconds",
            "Ictal_Segment_Count",
            "Correct_Ictal_Segments",
            "Event_Ictal_Sensitivity",
            "Mean_Ictal_Probability",
        ]
    ].sort_values(
        [
            "Case",
            "Event_ID",
        ]
    ).reset_index(drop=True)

    pooled_sensitivity = float(
        ictal_predictions[
            "Correct_Ictal_Prediction"
        ].mean()
    )

    median_event_sensitivity = float(
        event_results[
            "Event_Ictal_Sensitivity"
        ].median()
    )

    mean_event_sensitivity = float(
        event_results[
            "Event_Ictal_Sensitivity"
        ].mean()
    )

    long_event_all_segments = predictions[
        predictions["Event_ID"] == args.long_event_id
    ].copy()

    long_event_ictal = ictal_predictions[
        ictal_predictions["Event_ID"] == args.long_event_id
    ].copy()

    if len(long_event_all_segments) == 0:
        raise ValueError(
            f"Long event not found: {args.long_event_id}"
        )

    long_event_sensitivity = float(
        long_event_ictal[
            "Correct_Ictal_Prediction"
        ].mean()
    )

    without_long_event = ictal_predictions[
        ictal_predictions["Event_ID"]
        != args.long_event_id
    ].copy()

    sensitivity_without_long_event = float(
        without_long_event[
            "Correct_Ictal_Prediction"
        ].mean()
    )

    excluded_events = event_audit[
        event_audit["Status"] == "EXCLUDED"
    ].copy()

    exclusion_summary = (
        excluded_events["Exclusion_Reason"]
        .fillna("Reason not recorded")
        .value_counts()
        .rename_axis("Exclusion_Reason")
        .reset_index(name="Excluded_Event_Count")
    )

    summary = pd.DataFrame(
        [
            {
                "Metric": "Prespecified_Threshold",
                "Value": args.threshold,
                "Analysis_Type": "Primary",
            },
            {
                "Metric": "Included_Test_Events",
                "Value": len(event_results),
                "Analysis_Type": "Primary",
            },
            {
                "Metric": "Pooled_Ictal_Sensitivity",
                "Value": pooled_sensitivity,
                "Analysis_Type": "Primary",
            },
            {
                "Metric": "Median_Event_Ictal_Sensitivity",
                "Value": median_event_sensitivity,
                "Analysis_Type": "Required event analysis",
            },
            {
                "Metric": "Mean_Event_Ictal_Sensitivity",
                "Value": mean_event_sensitivity,
                "Analysis_Type": "Supplementary",
            },
            {
                "Metric": "Long_Event_ID",
                "Value": args.long_event_id,
                "Analysis_Type": "Event influence",
            },
            {
                "Metric": "Long_Event_EDF",
                "Value": long_event_all_segments[
                    "EDF_File"
                ].iloc[0],
                "Analysis_Type": "Event influence",
            },
            {
                "Metric": "Long_Event_Total_Matched_Segments",
                "Value": len(long_event_all_segments),
                "Analysis_Type": "Event influence",
            },
            {
                "Metric": "Long_Event_Ictal_Segments",
                "Value": len(long_event_ictal),
                "Analysis_Type": "Event influence",
            },
            {
                "Metric": "Long_Event_Ictal_Sensitivity",
                "Value": long_event_sensitivity,
                "Analysis_Type": "Event influence",
            },
            {
                "Metric": (
                    "Pooled_Ictal_Sensitivity_"
                    "Without_Long_Event"
                ),
                "Value": sensitivity_without_long_event,
                "Analysis_Type": "Post hoc",
            },
            {
                "Metric": "All_Audited_Events",
                "Value": len(event_audit),
                "Analysis_Type": "Audit",
            },
            {
                "Metric": "Included_Events",
                "Value": int(
                    (
                        event_audit["Status"]
                        == "INCLUDED"
                    ).sum()
                ),
                "Analysis_Type": "Audit",
            },
            {
                "Metric": "Excluded_Events",
                "Value": len(excluded_events),
                "Analysis_Type": "Audit",
            },
        ]
    )

    event_output = (
        output_folder
        / "chb_mit_event_analysis.csv"
    )

    summary_output = (
        output_folder
        / "chb_mit_event_analysis_summary.csv"
    )

    exclusion_output = (
        output_folder
        / "chb_mit_exclusion_summary.csv"
    )

    event_results.to_csv(
        event_output,
        index=False,
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    exclusion_summary.to_csv(
        exclusion_output,
        index=False,
    )

    print("=" * 72)
    print("CHB-MIT EVENT ANALYSIS COMPLETED")
    print("=" * 72)
    print("Threshold:", args.threshold)
    print("Test events:", len(event_results))
    print(
        "Pooled ictal sensitivity:",
        f"{pooled_sensitivity:.6f}",
    )
    print(
        "Median event sensitivity:",
        f"{median_event_sensitivity:.6f}",
    )
    print(
        "Long-event sensitivity:",
        f"{long_event_sensitivity:.6f}",
    )
    print(
        "Sensitivity without long event:",
        f"{sensitivity_without_long_event:.6f}",
    )
    print("Saved:", event_output)
    print("Saved:", summary_output)
    print("Saved:", exclusion_output)


if __name__ == "__main__":
    main()
