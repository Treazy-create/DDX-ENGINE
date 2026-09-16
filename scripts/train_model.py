import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3


def load_records(split):
    path = DATA / f"{split}.jsonl"
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_text(record, rng, partial=False, shuffle=False):
    # Preprocessing puts age and recorded sex in the first two lines.
    header = record["input_text"].splitlines()[:2]
    evidence = list(record["evidence"])

    initial = next(
        (
            item for item in evidence
            if item["code"] == record["initial_evidence"]
        ),
        evidence[0],
    )
    remaining = [item for item in evidence if item is not initial]

    if partial and remaining:
        # Keep the presenting complaint and a random subset of details.
        count = rng.randint(1, len(evidence) - 1)
        remaining = rng.sample(remaining, count - 1)
    elif shuffle:
        rng.shuffle(remaining)

    selected = [initial] + remaining
    lines = [
        f'{item["question"]} Answer: {item["answer"]}.'
        for item in selected
    ]
    return "\n".join(header + lines)


class EncodedCases(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.inputs.items()
        }
        item["labels"] = self.labels[index]
        return item


def make_loader(records, tokenizer, mode, seed, shuffle=False):
    rng = random.Random(seed)
    texts = []

    for record in records:
        partial = (
            rng.random() < 0.5
            if mode == "mixed"
            else mode == "partial"
        )
        texts.append(
            make_text(
                record,
                rng,
                partial=partial,
                shuffle=mode == "mixed",
            )
        )

    dataset = EncodedCases(
        texts,
        [record["label"] for record in records],
        tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def set_training_mode(model):
    model.train()

    # Keep frozen layers deterministic by disabling their dropout.
    model.distilbert.embeddings.eval()
    for layer in model.distilbert.transformer.layer[:-2]:
        layer.eval()


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    return output.loss.item()


@torch.inference_mode()
def evaluate(model, loader, class_count):
    model.eval()
    actual = []
    predicted = []

    for batch in loader:
        output = model(**batch)
        predicted.extend(output.logits.argmax(dim=-1).tolist())
        actual.extend(batch["labels"].tolist())

    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=list(range(class_count)),
            average="macro",
            zero_division=0,
        ),
    }


def benchmark(model, optimizer, train_loader, validation_loader):
    print("\nRunning 2 warm-up steps...", flush=True)
    iterator = iter(train_loader)
    set_training_mode(model)

    for _ in range(2):
        train_step(model, optimizer, next(iterator))

    print("Timing 10 training steps...", flush=True)
    start = time.perf_counter()

    for step in range(10):
        train_step(model, optimizer, next(iterator))
        print(f"  Timing step {step + 1}/10", flush=True)

    training_seconds = (time.perf_counter() - start) / 10

    model.eval()
    iterator = iter(validation_loader)

    with torch.inference_mode():
        model(**next(iterator))
        start = time.perf_counter()
        for _ in range(5):
            model(**next(iterator))
        validation_seconds = (time.perf_counter() - start) / 5

    epoch_seconds = (
        training_seconds * len(train_loader)
        + validation_seconds * len(validation_loader) * 2
    )

    print(f"\nTraining: {training_seconds:.2f} seconds per batch")
    print(f"Validation: {validation_seconds:.2f} seconds per batch")
    print(
        "Estimated 1 epoch including validation: "
        f"{epoch_seconds / 60:.1f} minutes"
    )
    print(
        f"Estimated {EPOCHS} epochs: "
        f"{epoch_seconds * EPOCHS / 3600:.2f} hours"
    )
    print("Excludes download, tokenization, and saving time.")
    print("Long runs may slow down as the CPU heats up.")
    print("Benchmark complete. Updated weights were NOT saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    mapping = json.loads(
        (DATA / "label_mapping.json").read_text(encoding="utf-8")
    )
    label2id = mapping["label2id"]
    id2label = {
        int(key): value
        for key, value in mapping["id2label"].items()
    }
    class_count = len(label2id)

    train_records = load_records("train")
    validation_records = load_records("validate")
    # The held-out test file is deliberately not loaded.

    print(f"Device: CPU | Threads: {torch.get_num_threads()}", flush=True)
    print(f"Training cases: {len(train_records):,}", flush=True)
    print(f"Validation cases: {len(validation_records):,}", flush=True)
    print(f"Conditions: {class_count}", flush=True)
    print(f"Loading {MODEL_NAME}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=class_count,
        label2id=label2id,
        id2label=id2label,
    )
    model.to("cpu")

    # Freeze all six transformer layers, then unfreeze the final two.
    # The newly initialized classification head remains trainable.
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False

    for layer in model.distilbert.transformer.layer[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    backbone_parameters = [
        parameter
        for parameter in model.distilbert.parameters()
        if parameter.requires_grad
    ]
    head_parameters = (
        list(model.pre_classifier.parameters())
        + list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 2e-5},
            {"params": head_parameters, "lr": 1e-4},
        ],
        weight_decay=0.01,
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}", flush=True)

    print("Preparing complete validation histories...", flush=True)
    validation_full = make_loader(
        validation_records, tokenizer, "full", SEED
    )

    if args.benchmark:
        train_loader = make_loader(
            train_records, tokenizer, "mixed", SEED, shuffle=True
        )
        benchmark(model, optimizer, train_loader, validation_full)
        return

    print("Preparing partial validation histories...", flush=True)
    validation_partial = make_loader(
        validation_records, tokenizer, "partial", SEED + 1000
    )

    run_name = "distilbert_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "models" / run_name
    output.mkdir(parents=True, exist_ok=False)

    report = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs_planned": EPOCHS,
        "train_cases": len(train_records),
        "validation_cases": len(validation_records),
        "trainable_parameters": trainable,
        "training_strategy": "last_two_layers_and_classification_head",
        "backbone_learning_rate": 2e-5,
        "head_learning_rate": 1e-4,
        "selection_metric": "mean_full_and_partial_validation_macro_f1",
        "test_used": False,
        "history": [],
    }

    (output / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    best_score = -1.0
    overall_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nPreparing epoch {epoch}/{EPOCHS}...", flush=True)
        train_loader = make_loader(
            train_records,
            tokenizer,
            "mixed",
            SEED + epoch,
            shuffle=True,
        )

        set_training_mode(model)
        loss_total = 0.0
        examples_seen = 0
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            loss = train_step(model, optimizer, batch)
            size = len(batch["labels"])
            loss_total += loss * size
            examples_seen += size

            if step == 1 or step % 25 == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - epoch_start
                remaining = elapsed / step * (len(train_loader) - step)
                print(
                    f"Epoch {epoch} | batch {step}/{len(train_loader)} "
                    f"| loss {loss_total / examples_seen:.4f} "
                    f"| training time left ~{remaining / 60:.1f} min",
                    flush=True,
                )

        print("Validating complete histories...", flush=True)
        full_metrics = evaluate(model, validation_full, class_count)

        print("Validating partial histories...", flush=True)
        partial_metrics = evaluate(model, validation_partial, class_count)

        score = (
            full_metrics["macro_f1"] + partial_metrics["macro_f1"]
        ) / 2

        epoch_result = {
            "epoch": epoch,
            "training_loss": loss_total / examples_seen,
            "full_validation": full_metrics,
            "partial_validation": partial_metrics,
            "selection_score": score,
            "seconds": time.perf_counter() - epoch_start,
        }
        report["history"].append(epoch_result)
        print(json.dumps(epoch_result, indent=2), flush=True)

        if score > best_score:
            best_score = score
            model.save_pretrained(output / "best_model")
            tokenizer.save_pretrained(output / "best_model")
            report["best_epoch"] = epoch
            report["best_score"] = score
            print("Saved new best model.", flush=True)

        report["elapsed_seconds"] = time.perf_counter() - overall_start
        (output / "training_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    print(f"\nTraining complete. Results: {output}", flush=True)
    print(f"Best epoch: {report['best_epoch']}", flush=True)
    print("Test data remains untouched. No OpenAI credits used.")


if __name__ == "__main__":
    main()
