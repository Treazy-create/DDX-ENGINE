"""BERT fine-tuning and atomic, resumable local checkpoints."""
import json
import time

import torch

from classifier_data import evaluate, loader


def configure_training(model, layers):
    if not 1 <= layers <= len(model.bert.encoder.layer):
        raise ValueError('Trainable layer count is outside this model architecture')
    for parameter in model.bert.parameters():
        parameter.requires_grad = False
    for layer in model.bert.encoder.layer[-layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if model.bert.pooler is not None:
        for parameter in model.bert.pooler.parameters():
            parameter.requires_grad = True
    return torch.optim.AdamW([
        {'params': [p for p in model.bert.parameters() if p.requires_grad], 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}], weight_decay=.01)


def training_mode(model, layers):
    model.train()
    model.bert.embeddings.eval()
    for layer in model.bert.encoder.layer[:-layers]:
        layer.eval()


def step(model, optimizer, batch, device):
    optimizer.zero_grad(set_to_none=True)
    loss = model(**{key: value.to(device) for key, value in batch.items()}).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
    optimizer.step()
    return loss.item()


def save_checkpoint(folder, model, optimizer, state):
    target = folder / 'last_checkpoint.pt'
    temporary = target.with_suffix('.tmp')
    torch.save({**state, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': model.config.to_dict(), 'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary)
    temporary.replace(target)


def train(model, tokenizer, optimizer, rows, validation, args, folder, state, device):
    full = loader(validation, tokenizer, 'full', args.seed, args.batch_size, args.max_length)
    partial = loader(validation, tokenizer, 'partial', args.seed + 1000, args.batch_size, args.max_length)
    for epoch in range(state['epoch'], args.epochs + 1):
        batches = loader(rows, tokenizer, 'mixed', args.seed + epoch, args.batch_size, args.max_length)
        completed = state['step'] if epoch == state['epoch'] else 0
        training_mode(model, args.layers)
        started = time.perf_counter()
        try:
            for index, batch in enumerate(batches, 1):
                if index <= completed:
                    continue
                loss = step(model, optimizer, batch, device)
                state.update(epoch=epoch, step=index)
                if index == 1 or index % 25 == 0:
                    seconds = (time.perf_counter() - started) / (index - completed)
                    print(f'Epoch {epoch}/{args.epochs} | batch {index}/{len(batches)} | '
                          f'loss {loss:.4f} | remaining training ~{seconds * (len(batches)-index)/60:.1f} min', flush=True)
                if index % args.save_every == 0:
                    save_checkpoint(folder, model, optimizer, state)
        except KeyboardInterrupt:
            # Interrupts may arrive halfway through an optimizer update. Resume
            # from the last atomic save instead of persisting a partial step.
            print(f'Stopped. Resume the last saved checkpoint with --resume "{folder}"', flush=True)
            return
        print('Validating full histories...', flush=True)
        full_metrics = evaluate(model, full, device, model.config.num_labels)
        print('Validating partial histories...', flush=True)
        partial_metrics = evaluate(model, partial, device, model.config.num_labels)
        score = (full_metrics['macro_f1'] + partial_metrics['macro_f1']) / 2
        result = {'epoch': epoch, 'full_validation': full_metrics, 'partial_validation': partial_metrics}
        state['history'].append(result)
        if score > state['best_score']:
            model.save_pretrained(folder / 'best_model')
            tokenizer.save_pretrained(folder / 'best_model')
            state.update(best_score=score, best_epoch=epoch)
        state.update(epoch=epoch + 1, step=0)
        save_checkpoint(folder, model, optimizer, state)
        report = {key: value for key, value in state.items()}
        (folder / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
    print(f'Training complete. Best model: {folder / "best_model"}', flush=True)
    print('Test data untouched. Compare validation results before changing MODEL_PATH.')
