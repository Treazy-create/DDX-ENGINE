# Stage two: Bio_ClinicalBERT classifier experiment

This stage adds a training and comparison pipeline. It does not replace the live
model automatically. Bio_ClinicalBERT is an encoder fine-tuned here for sequence
classification, not a conversational generator. Its pretraining includes clinical
notes, but that does not guarantee improvement on this task.

Source: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

## Commands (project root)

First, time a few real training batches. This downloads model/tokenizer files on
first use, makes no OpenAI calls, and does not save the benchmark's updated weights:

```powershell
python scripts/train_clinicalbert.py --benchmark
```

Then launch training (CPU default, three epochs, final two BERT layers plus pooler
and classification head). Preserve the printed run folder:

```powershell
python scripts/train_clinicalbert.py
```

Resume after interruption, substituting the actual run folder:

```powershell
python scripts/train_clinicalbert.py --resume "models/clinicalbert_YYYYMMDD_HHMMSS"
```

Checkpoints save every 100 completed batches and at epoch boundaries. Sudden
shutdown loses work since the last completed checkpoint. Ctrl+C also resumes
from that checkpoint, not an optimizer step that may have been interrupted.
Resume requires unchanged training data and training settings. Validation may
rerun after interruption; it never trains the model. Keep the same device for
reproducibility. Internet is needed on first download, not for cached training.
The estimate excludes validation/checkpoint saving and is not a completion promise.

## Compare before switching

Pass both actual model folders to the evaluator; it defaults to validation data
and identical full and partial histories, matching the baseline's partial seed:

```powershell
python scripts/evaluate_classifier.py --models "models/distilbert_20260916_013654/best_model" "models/clinicalbert_YYYYMMDD_HHMMSS/best_model"
```

After selecting the model using validation, evaluate the selected model on the
held-out test set using `--split test`. Do not tune on test results. Reports are
saved in reports/classifier_evaluation. Synthetic accuracy is not clinical accuracy.

Only after comparing results should MODEL_PATH in backend/.env point to the actual
new best_model folder. Restart the backend afterward. The existing backend already
uses AutoModelForSequenceClassification, so no architecture-specific rewrite is
needed. Never place an API key in frontend files or commit backend/.env.

## Scope and resources

The same processed 49-condition pilot dataset is used. This does not add malaria,
typhoid, or the nursing reference's conditions to the training labels. Reference
lists are not symptom-labelled cases. No new disease coverage is claimed.

CPU training can take hours and may be slower than DistilBERT. The script requires
at least 3 GB free at startup; keep additional headroom for cache and temporary
atomic saves. One resumable checkpoint and a best model are kept per run. Each new
run uses additional disk. GPU execution is optional with --device cuda if available.

No dependency upgrades are installed automatically. Use the working project's
torch, transformers, and scikit-learn environment. Tiny offline tests exercise
actual gradients and checkpoint restoration; full pretrained training and resulting
accuracy remain unverified until you run them.
