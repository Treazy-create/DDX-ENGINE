"""Compare saved classifiers on identical histories; validation is the default."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier_data import ROOT, evaluate, load_records, loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', type=Path, required=True)
    parser.add_argument('--split', choices=['validate', 'test'], default='validate')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or not 8 <= args.max_length <= 512:
        parser.error('Positive batch-size required; max-length must be 8..512')
    torch.set_num_threads(4)
    data = ROOT / 'data/processed'
    mapping = json.loads((data / 'label_mapping.json').read_text(encoding='utf-8'))['label2id']
    rows = load_records(data, args.split)
    results = []
    for path in args.models:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, pad_token='[PAD]')
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        if model.config.label2id != mapping:
            raise ValueError(f'{path}: label mapping does not match this dataset')
        result = {'model': str(path), 'architecture': model.config.model_type, 'split': args.split}
        for mode, seed in [('full', 42), ('partial', 1042)]:
            print(f'Evaluating {path.name}: {mode} histories...', flush=True)
            batches = loader(rows, tokenizer, mode, seed, args.batch_size, args.max_length)
            result[mode] = evaluate(model, batches, 'cpu', len(mapping))
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        del model
    folder = ROOT / 'reports/classifier_evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    target.write_text(json.dumps({'max_length': args.max_length, 'partial_seed': 1042,
                                 'clinical_validation': False, 'results': results}, indent=2), encoding='utf-8')
    print(f'Saved {target}. These results measure synthetic cases, not clinical accuracy.')


if __name__ == '__main__':
    main()
