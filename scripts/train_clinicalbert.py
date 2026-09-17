"""Train Bio_ClinicalBERT as a classifier; never overwrite the DistilBERT baseline."""
import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

from classifier_data import ROOT, dataset_fingerprint, load_records, loader
from classifier_training import configure_training, save_checkpoint, step, train, training_mode

MODEL = 'emilyalsentzer/Bio_ClinicalBERT'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true', help='Time 5 batches; do not save trained weights')
    parser.add_argument('--resume', type=Path, help='Existing run folder with last_checkpoint.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.layers, args.save_every) < 1 or not 8 <= args.max_length <= 512:
        parser.error('Positive counts required; max-length must be 8..512')
    if args.resume and (args.benchmark or args.output):
        parser.error('--resume cannot be combined with --benchmark or --output')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('No CUDA GPU available; use --device cpu')
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        parser.error('Free at least 3 GB for weights, checkpoints and temporary saves')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    data = ROOT / 'data/processed'
    mapping = json.loads((data / 'label_mapping.json').read_text(encoding='utf-8'))
    rows, validation = load_records(data, 'train'), load_records(data, 'validate')
    fingerprint = dataset_fingerprint(data)
    settings = {key: getattr(args, key) for key in ('batch_size', 'max_length', 'layers', 'seed')}
    folder = args.resume or args.output or ROOT / 'models' / ('clinicalbert_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(folder / 'last_checkpoint.pt', map_location='cpu', weights_only=True)
        if checkpoint['data_fingerprint'] != fingerprint or checkpoint['settings'] != settings:
            parser.error('Resume data/settings differ. Use the same data, batch-size, max-length, layers and seed.')
        model = AutoModelForSequenceClassification.from_config(BertConfig.from_dict(checkpoint['config']))
        model.load_state_dict(checkpoint['model'])
        tokenizer = AutoTokenizer.from_pretrained(folder / 'tokenizer', local_files_only=True)
    else:
        if folder.exists():
            parser.error('Output folder already exists. Choose a new folder or use --resume.')
        print(f'Loading {MODEL}; the first run requires internet.', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, pad_token='[PAD]', unk_token='[UNK]',
                                                 cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=len(mapping['label2id']), label2id=mapping['label2id'],
            id2label={int(k): v for k, v in mapping['id2label'].items()})
    model.to(args.device)
    optimizer = configure_training(model, args.layers)
    state = {'epoch': 1, 'step': 0, 'history': [], 'best_score': -1., 'settings': settings,
             'data_fingerprint': fingerprint, 'base_model': MODEL, 'test_used': False,
             'revision': getattr(model.config, '_commit_hash', None)}
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        state = {k: v for k, v in checkpoint.items() if k not in ('model', 'optimizer', 'config', 'rng', 'cuda_rng')}
        torch.set_rng_state(checkpoint['rng'])
        if args.device == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    print(f'Device: {args.device} | Training: {len(rows)} | Validation: {len(validation)} | '
          f'Conditions: {len(mapping["label2id"])}', flush=True)
    if args.benchmark:
        batches = loader(rows, tokenizer, 'mixed', args.seed + 1, args.batch_size, args.max_length)
        training_mode(model, args.layers)
        iterator = iter(batches)
        step(model, optimizer, next(iterator), args.device)
        started = time.perf_counter()
        for _ in range(min(5, len(batches) - 1)):
            step(model, optimizer, next(iterator), args.device)
        measured = min(5, len(batches) - 1)
        if measured:
            seconds = (time.perf_counter() - started) / measured
            print(f'{seconds:.1f} seconds/batch; training-only estimate: '
                  f'{seconds * len(batches) * args.epochs / 3600:.2f} hours. Validation/saving add time.')
        print('Benchmark only: updated weights were not saved.')
        return
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(folder / 'tokenizer')
    if checkpoint is None:
        save_checkpoint(folder, model, optimizer, state)
    print(f'Run folder: {folder}', flush=True)
    train(model, tokenizer, optimizer, rows, validation, args, folder, state, args.device)


if __name__ == '__main__':
    main()
