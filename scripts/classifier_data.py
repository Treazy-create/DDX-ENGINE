"""Shared, reproducible full/partial histories for training and evaluation."""
import hashlib
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


def load_records(folder, split):
    with (folder / f'{split}.jsonl').open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or any(not row.get('evidence') for row in rows):
        raise ValueError(f'{split}: empty dataset or case without evidence')
    return rows


def dataset_fingerprint(folder):
    digest = hashlib.sha256()
    for filename in ('train.jsonl', 'validate.jsonl', 'label_mapping.json'):
        digest.update(filename.encode())
        with (folder / filename).open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def case_text(row, rng, partial=False, shuffle=False):
    evidence = list(row['evidence'])
    initial = next((item for item in evidence if item['code'] == row['initial_evidence']), evidence[0])
    remaining = [item for item in evidence if item is not initial]
    if partial and remaining:
        remaining = rng.sample(remaining, rng.randint(1, len(evidence) - 1) - 1)
    elif shuffle:
        rng.shuffle(remaining)
    return '\n'.join(row['input_text'].splitlines()[:2] +
                     [f"{item['question']} Answer: {item['answer']}." for item in [initial] + remaining])


class Cases(Dataset):
    def __init__(self, rows, tokenizer, mode, seed, max_length):
        rng = random.Random(seed)
        texts = [case_text(row, rng, rng.random() < .5 if mode == 'mixed' else mode == 'partial',
                           mode == 'mixed') for row in rows]
        self.inputs = tokenizer(texts, padding='max_length', truncation=True,
                                max_length=max_length, return_tensors='pt', return_token_type_ids=False)
        self.labels = torch.tensor([row['label'] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: value[index] for key, value in self.inputs.items()}, 'labels': self.labels[index]}


def loader(rows, tokenizer, mode, seed, batch_size, max_length):
    return DataLoader(Cases(rows, tokenizer, mode, seed, max_length), batch_size=batch_size,
                      shuffle=mode == 'mixed', num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def evaluate(model, batches, device, count):
    model.eval()
    expected, predicted = [], []
    for batch in batches:
        result = model(**{key: value.to(device) for key, value in batch.items()})
        expected.extend(batch['labels'].tolist())
        predicted.extend(result.logits.argmax(-1).cpu().tolist())
    return {'accuracy': accuracy_score(expected, predicted),
            'macro_f1': f1_score(expected, predicted, labels=list(range(count)), average='macro', zero_division=0),
            'cases': len(expected)}
