from pathlib import Path
import argparse
import subprocess
import urllib.request


BASE_URL = (
    "https://physionet.org/files/chbmit/1.0.0/"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download selected CHB-MIT cases "
            "from PhysioNet."
        )
    )

    parser.add_argument(
        "--output_folder",
        required=True
    )

    parser.add_argument(
        "--first_case",
        type=int,
        default=1
    )

    parser.add_argument(
        "--last_case",
        type=int,
        default=12
    )

    args = parser.parse_args()

    if (
        args.first_case < 1
        or args.last_case > 24
        or args.first_case > args.last_case
    ):
        raise ValueError(
            "Cases must be between 1 and 24."
        )

    output_folder = Path(
        args.output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    records_url = BASE_URL + "RECORDS"

    print("Reading PhysioNet file list...")

    with urllib.request.urlopen(
        records_url
    ) as response:
        records = (
            response.read()
            .decode("utf-8")
            .splitlines()
        )

    selected_cases = [
        f"chb{number:02d}"
        for number in range(
            args.first_case,
            args.last_case + 1
        )
    ]

    print("Cases:", selected_cases)
    print("Output:", output_folder)

    for case_name in selected_cases:
        case_folder = (
            output_folder / case_name
        )

        case_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        edf_records = [
            record.strip()
            for record in records
            if record.strip().startswith(
                case_name + "/"
            )
            and record.strip().endswith(".edf")
        ]

        files_to_download = (
            edf_records
            + [
                f"{case_name}/"
                f"{case_name}-summary.txt"
            ]
        )

        print(
            f"\n{case_name}: "
            f"{len(edf_records)} EDF files"
        )

        for relative_path in files_to_download:
            file_name = Path(
                relative_path
            ).name

            destination = (
                case_folder / file_name
            )

            url = BASE_URL + relative_path

            command = [
                "wget",
                "-c",
                "-q",
                "--show-progress",
                "-O",
                str(destination),
                url,
            ]

            result = subprocess.run(
                command
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Download failed: {url}"
                )

        print(f"{case_name}: completed")

    print("\nDOWNLOAD COMPLETED")
    print("Saved to:", output_folder)


if __name__ == "__main__":
    main()
