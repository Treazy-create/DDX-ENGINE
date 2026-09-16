import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_model import SEED, load_records, make_loader


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models" / "distilbert_20260916_013654"
MODEL = RUN / "best_model"
OUTPUT = ROOT / "reports" / RUN.name


@torch.inference_mode()
def assess(model, loader, names, preview=False):
    actual = []
    predicted = []
    started = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        labels = batch.pop("labels")
        logits = model(**batch).logits
        predictions = logits.argmax(dim=-1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.tolist())

        if preview:
            for expected, result in zip(labels.tolist(), predictions.tolist()):
                print(f"Expected:  {names[expected]}")
                print(f"Predicted: {names[result]}")
                print()

        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f"Checked batch {step}/{len(loader)}", flush=True)

    return {
        "cases": len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=list(range(len(names))),
            average="macro",
            zero_division=0,
        ),
        "seconds": time.perf_counter() - started,
        "per_condition": classification_report(
            actual,
            predicted,
            labels=list(range(len(names))),
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(SEED)

    mapping = json.loads(
        (RUN / "label_mapping.json").read_text(encoding="utf-8")
    )
    names = [
        mapping["id2label"][str(index)]
        for index in range(len(mapping["id2label"]))
    ]

    print("Loading your saved model locally...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()

    for index, name in enumerate(names):
        if model.config.id2label[index] != name:
            raise ValueError("Model and dataset label mappings do not match.")

    # Quick mode uses validation cases, leaving the test set untouched.
    records = load_records("validate" if args.quick else "test")
    if args.quick:
        records = records[:8]

    print(f"Model loaded. Conditions: {len(names)}", flush=True)
    print(f"Cases to check: {len(records):,}", flush=True)

    results = {
        "model_path": str(MODEL),
        "split": "validation_smoke_check" if args.quick else "test",
        "partial_history_seed": SEED + 2000,
        "results": {},
    }

    modes = ["full"] if args.quick else ["full", "partial"]

    for mode in modes:
        print(f"\nChecking {mode} histories...", flush=True)
        loader = make_loader(
            records,
            tokenizer,
            mode,
            SEED + 2000,
        )
        metrics = assess(model, loader, names, preview=args.quick)
        results["results"][mode] = metrics

        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Macro-F1: {metrics['macro_f1']:.2%}")

        if not args.quick:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            (OUTPUT / "test_results.json").write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )

    if args.quick:
        print("\nLoading and prediction check complete.")
        print("Eight examples are not a final accuracy estimate.")
    else:
        print(f"\nTest report saved: {OUTPUT / 'test_results.json'}")

    print("No training performed. No OpenAI credits used.")


if __name__ == "__main__":
    main()
