import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REQUIRED_FILES = [
    "train.csv",
    "validate.csv",
    "test.csv",
    "release_conditions.json",
    "release_evidences.json",
]


def load_json(filename):
    path = RAW_DIR / filename

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def inspect_csv(filename):
    path = RAW_DIR / filename
    condition_counts = Counter()
    missing_diagnosis = 0
    missing_evidence = 0
    sample_rows = []
    row_count = 0

    print(f"\nReading {filename}...", flush=True)

    # Read one row at a time instead of loading the whole file into RAM.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        required_columns = {"PATHOLOGY", "EVIDENCES"}
        missing_columns = required_columns - set(columns)

        if missing_columns:
            raise ValueError(
                f"{filename} is missing expected columns: "
                f"{sorted(missing_columns)}. Found: {columns}"
            )

        for row in reader:
            row_count += 1

            diagnosis = (row.get("PATHOLOGY") or "").strip()
            evidence = (row.get("EVIDENCES") or "").strip()

            if diagnosis:
                condition_counts[diagnosis] += 1
            else:
                missing_diagnosis += 1

            if not evidence or evidence in ("[]", "null", "None"):
                missing_evidence += 1

            if len(sample_rows) < 2:
                sample_rows.append(row)

            if row_count % 200_000 == 0:
                print(f"  Read {row_count:,} rows...", flush=True)

    result = {
        "filename": filename,
        "size_mb": round(path.stat().st_size / 1_000_000, 2),
        "rows": row_count,
        "columns": columns,
        "unique_conditions": len(condition_counts),
        "missing_diagnosis": missing_diagnosis,
        "missing_or_empty_evidence": missing_evidence,
        "condition_counts": dict(sorted(condition_counts.items())),
        "sample_rows": sample_rows,
    }

    print(f"  Rows: {row_count:,}")
    print(f"  Conditions: {len(condition_counts)}")
    print(f"  Missing diagnoses: {missing_diagnosis}")
    print(f"  Missing/empty evidence: {missing_evidence}")
    print(f"  Columns: {', '.join(columns)}")

    return result


def describe_metadata(name, data):
    print(f"\n{name}")
    print(f"  Structure: {type(data).__name__}")
    print(f"  Entries: {len(data):,}")

    if isinstance(data, dict) and data:
        first_key = next(iter(data))
        print(f"  First key: {first_key}")
        print(json.dumps(data[first_key], indent=2, ensure_ascii=False))

    elif isinstance(data, list) and data:
        print(json.dumps(data[0], indent=2, ensure_ascii=False))


def main():
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (RAW_DIR / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing files in {RAW_DIR}:\n"
            + "\n".join(f"  - {filename}" for filename in missing)
        )

    print("DDXPlus dataset inspection")
    print(f"Data folder: {RAW_DIR}")
    print("Raw files will not be changed.")

    conditions = load_json("release_conditions.json")
    evidences = load_json("release_evidences.json")

    describe_metadata("Condition definitions", conditions)
    describe_metadata("Symptom/history definitions", evidences)

    splits = {
        name: inspect_csv(f"{name}.csv")
        for name in ("train", "validate", "test")
    }

    train_labels = set(splits["train"]["condition_counts"])

    unseen_labels = {
        name: sorted(
            set(splits[name]["condition_counts"]) - train_labels
        )
        for name in ("validate", "test")
    }

    print("\nConditions absent from the training split:")
    for name, labels in unseen_labels.items():
        print(f"  {name}: {labels if labels else 'None'}")

    print("\nFirst training example:")
    samples = splits["train"]["sample_rows"]

    if samples:
        for column, value in samples[0].items():
            print(f"  {column}: {value}")

    report = {
        "condition_definition_count": len(conditions),
        "evidence_definition_count": len(evidences),
        "splits": splits,
        "labels_absent_from_training": unseen_labels,
        "note": (
            "Initial structure and completeness inspection only. "
            "Duplicate checks, evidence decoding, and leakage checks "
            "will be performed during preprocessing."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "dataset_inspection.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {report_path}")
    print("Inspection complete.")


if __name__ == "__main__":
    main()
