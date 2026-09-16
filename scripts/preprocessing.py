import ast
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEED = 42

# Small first-run datasets. Increase only after the pipeline works.
LIMITS = {
    "train": 200,
    "validate": 50,
    "test": 50,
}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def parse_evidence(raw_value):
    # DDXPlus stores Python-style lists inside CSV cells.
    # literal_eval reads these safely; never use eval().
    value = ast.literal_eval(raw_value)

    if not isinstance(value, list):
        raise ValueError("EVIDENCES must contain a list.")

    if not value or not all(isinstance(item, str) for item in value):
        raise ValueError("EVIDENCES must contain non-empty string entries.")

    return sorted(set(value))


def decode_evidence(items, definitions):
    grouped = defaultdict(list)

    for item in items:
        code, separator, value = item.partition("_@_")

        if code not in definitions:
            raise ValueError(f"Unknown evidence code: {code}")

        metadata = definitions[code]
        evidence_type = metadata["data_type"]

        if evidence_type == "B":
            if separator:
                raise ValueError(
                    f"Unexpected value attached to binary evidence: {item}"
                )

            answer = "Yes"

        elif evidence_type in {"C", "M"}:
            if not separator:
                raise ValueError(
                    f"Missing value for categorical evidence: {item}"
                )

            meanings = metadata.get("value_meaning", {})
            meaning = meanings.get(value)

            if isinstance(meaning, dict):
                answer = meaning.get("en")
            elif isinstance(meaning, str):
                answer = meaning
            else:
                # Some values are numeric categories rather than V_ codes.
                if value.startswith("V_"):
                    raise ValueError(f"No English meaning found for {item}")

                answer = value

            if answer is None or str(answer).strip() == "":
                raise ValueError(f"Empty answer meaning for {item}")

        else:
            raise ValueError(
                f"Unsupported evidence type {evidence_type!r} for {code}"
            )

        grouped[code].append(str(answer))

    decoded = []

    for code in sorted(grouped):
        question = definitions[code].get("question_en", "").strip()

        if not question:
            raise ValueError(f"Missing English question for {code}")

        answers = sorted(set(grouped[code]))

        decoded.append({
            "code": code,
            "question": question,
            "answer": "; ".join(answers),
            "is_history": bool(
                definitions[code].get("is_antecedent", False)
            ),
        })

    return decoded


def build_record(row, definitions, label2id, fingerprint):
    evidence_items = parse_evidence(row["EVIDENCES"])
    decoded = decode_evidence(evidence_items, definitions)

    age = int(row["AGE"])
    sex_code = row["SEX"].strip()
    sex = {"M": "male", "F": "female"}.get(sex_code, sex_code)

    initial_code = row["INITIAL_EVIDENCE"].strip().partition("_@_")[0]

    # Put the opening complaint first without repeating it.
    decoded.sort(
        key=lambda item: (
            item["code"] != initial_code,
            item["code"],
        )
    )

    text_lines = [
        f"Age: {age} years.",
        f"Recorded sex: {sex}.",
    ]

    text_lines.extend(
        f"{item['question']} Answer: {item['answer']}."
        for item in decoded
    )

    diagnosis = row["PATHOLOGY"].strip()

    return {
        "case_id": fingerprint,
        "input_text": "\n".join(text_lines),
        "label": label2id[diagnosis],
        "diagnosis": diagnosis,
        "initial_evidence": initial_code,
        "evidence": decoded,
    }


def process_split(
    split,
    limit,
    definitions,
    label2id,
    seen_cases,
    seed,
):
    rng = random.Random(seed)
    pools = defaultdict(list)
    unique_counts = Counter()

    stats = {
        "rows_read": 0,
        "duplicates_within_split": 0,
        "duplicates_from_earlier_splits": 0,
        "conflicting_labels": 0,
    }

    print(f"\nProcessing {split}.csv...", flush=True)

    with (RAW / f"{split}.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            stats["rows_read"] += 1
            diagnosis = row["PATHOLOGY"].strip()

            if diagnosis not in label2id:
                raise ValueError(f"Unknown diagnosis: {diagnosis}")

            evidence_items = parse_evidence(row["EVIDENCES"])

            # Compare cases using patient information, NOT the answer.
            # INITIAL_EVIDENCE is excluded because the same case can
            # start the conversation with different complaints.
            identity = json.dumps(
                [
                    int(row["AGE"]),
                    row["SEX"].strip(),
                    evidence_items,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )

            digest = hashlib.sha256(identity.encode("utf-8")).digest()
            previous = seen_cases.get(digest)

            if previous is not None:
                previous_label, previous_split = previous

                if previous_label != label2id[diagnosis]:
                    stats["conflicting_labels"] += 1

                if previous_split == split:
                    stats["duplicates_within_split"] += 1
                else:
                    stats["duplicates_from_earlier_splits"] += 1

            else:
                seen_cases[digest] = (label2id[diagnosis], split)
                unique_counts[diagnosis] += 1
                number_seen = unique_counts[diagnosis]

                # Reservoir sampling: randomly sample each condition
                # without holding the whole CSV in memory.
                if len(pools[diagnosis]) < limit:
                    position = len(pools[diagnosis])
                else:
                    position = rng.randrange(number_seen)

                if position < limit:
                    record = build_record(
                        row,
                        definitions,
                        label2id,
                        digest.hex(),
                    )

                    if position == len(pools[diagnosis]):
                        pools[diagnosis].append(record)
                    else:
                        pools[diagnosis][position] = record

            if stats["rows_read"] % 200_000 == 0:
                print(
                    f"  Checked {stats['rows_read']:,} rows...",
                    flush=True,
                )

    records = [
        record
        for diagnosis in sorted(pools)
        for record in pools[diagnosis]
    ]
    rng.shuffle(records)

    stats["unique_cases"] = sum(unique_counts.values())
    stats["selected_cases"] = len(records)
    stats["selected_per_condition"] = {
        diagnosis: len(pools[diagnosis])
        for diagnosis in sorted(label2id)
    }

    print(f"  Selected cases: {len(records):,}")
    print(
        "  Duplicate cases:",
        stats["duplicates_within_split"]
        + stats["duplicates_from_earlier_splits"],
    )
    print(f"  Conflicting labels: {stats['conflicting_labels']}")

    return records, stats


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    conditions = read_json(RAW / "release_conditions.json")
    definitions = read_json(RAW / "release_evidences.json")

    condition_names = sorted(
        metadata["condition_name"]
        for metadata in conditions.values()
    )

    if len(condition_names) != len(set(condition_names)):
        raise ValueError("Condition definitions contain duplicate names.")

    label2id = {
        name: index
        for index, name in enumerate(condition_names)
    }

    id2label = {
        str(index): name
        for name, index in label2id.items()
    }

    seen_cases = {}
    selected_splits = {}
    split_reports = {}

    for index, (split, limit) in enumerate(LIMITS.items()):
        records, stats = process_split(
            split=split,
            limit=limit,
            definitions=definitions,
            label2id=label2id,
            seen_cases=seen_cases,
            seed=SEED + index,
        )

        selected_splits[split] = records
        split_reports[split] = stats

    report = {
        "seed": SEED,
        "per_condition_limits": LIMITS,
        "splits": split_reports,
        "notes": [
            "Synthetic patient cases, not clinical validation.",
            "Missing evidence is not converted into negative evidence.",
            "Duplicate identity uses age, sex, and sorted evidence.",
            "Earlier splits take precedence: train, validate, test.",
            "Differential diagnoses are excluded from model inputs.",
            "These are sampled subsets, not full benchmark evaluation.",
        ],
    }

    report_path = REPORTS / "preprocessing_report.json"
    write_json(report_path, report)

    conflicts = sum(
        stats["conflicting_labels"]
        for stats in split_reports.values()
    )

    if conflicts:
        raise ValueError(
            f"Found {conflicts} identical patient descriptions with "
            "different diagnosis labels. The report was saved, but "
            "new processed datasets were not written. Share this "
            "result before proceeding."
        )

    for split, records in selected_splits.items():
        missing_labels = [
            name
            for name, count in
            split_reports[split]["selected_per_condition"].items()
            if count == 0
        ]

        if missing_labels:
            raise ValueError(
                f"{split} has no selected cases for: {missing_labels}"
            )

        output_path = PROCESSED / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        PROCESSED / "label_mapping.json",
        {"label2id": label2id, "id2label": id2label},
    )

    # Preserve definitions for later use by the conversational backend.
    write_json(PROCESSED / "evidence_definitions.json", definitions)
    write_json(PROCESSED / "condition_definitions.json", conditions)

    print("\nExample processed training input:")
    print(selected_splits["train"][0]["input_text"])

    print("\nDataset sizes:")
    for split, records in selected_splits.items():
        print(f"  {split}: {len(records):,}")

    print(f"\nSaved processed files to: {PROCESSED}")
    print(f"Saved report to: {report_path}")
    print("Preprocessing complete. Raw files were not changed.")


if __name__ == "__main__":
    main()
