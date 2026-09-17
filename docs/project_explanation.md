# Project explanation and defence notes

## Problem and engineering contribution

The original interface accepted a symptom description and returned ranked labels.
It did not adequately support follow-up dialogue. The upgrade separates natural
language handling, evidence state, local inference and user interaction. The
project's contribution includes data preparation, fine-tuning, evaluation,
conversation orchestration, voice lifecycle and frontend integration. It uses
pretrained models and hosted APIs; it is not an LLM created from scratch.

## Data preparation

The inspected English DDXPlus release had 1,025,602 training rows, 132,448 validation
rows, and 134,529 test rows. All splits covered 49 conditions. There were no missing
diagnosis/empty-evidence fields in that inspection. Deduplication reported 29,823
training, 4,619 validation, and 5,452 test duplicates, with no conflicting labels.

The pilot selected up to 200 training cases and 50 validation/test cases per class,
giving 9,800 / 2,426 / 2,434 examples. Seed 42 controls sampling. Labels and supplied
differential-diagnosis lists are not inserted into input text. Symptom codes become
question/answer text, with age and recorded sex. Partial histories simulate earlier
interview stages. Synthetic cases do not establish real-world medical performance.

## Completed DistilBERT baseline

The reported three-epoch CPU run trained the final two transformer layers and
classification head: 14,804,017 trainable parameters out of 66,991,153. Batch size 8,
maximum length 256, encoder learning rate 2e-5 and head rate 1e-4. It mixed full and
partial histories and selected the best epoch using validation macro-F1.

Reported best epoch: 3. Elapsed time: approximately 2 hours 29 minutes.

| Validation view | Accuracy | Macro-F1 |
|---|---:|---:|
| Full history | 0.98887 | 0.98845 |
| Partial history | 0.86232 | 0.86658 |

These are the existing baseline's synthetic **validation** results, not held-out
test or clinical accuracy. The completed comparison and held-out test results are reported below.

## Bio_ClinicalBERT experiment

The replacement candidate is `emilyalsentzer/Bio_ClinicalBERT`, pretrained using
biomedical text and clinical notes. The new pipeline fine-tunes the final two BERT
layers, pooler, and classifier by default. It saves optimizer/model/random state
and a dataset/settings fingerprint so training can resume. It retains the same
pilot labels; a different encoder does not add malaria or other absent conditions.

Both candidates must be evaluated on identical full/partial validation histories.
Choose based on performance, CPU response time, and robustness rather than the
model name. Evaluate the selected model on held-out data afterward.

## Dialogue and voice

The classifier supplies the possible condition. OpenAI handles structured evidence
extraction and optional phrasing, not independent diagnostic labels. The symptom
state records positive/negative/unknown findings and accepts corrections. The UI
remains usable after assessment rather than requiring a new chat for every question.

Voice mode uses an explicit Send boundary. It stops listening while speaking to
reduce self-transcription. Browser recognition gives live captions where available;
the recording option uses OpenAI transcription after Send. Speech output is browser
synthesis. These are different from a realtime multimodal speech-to-speech API.

## Honest limitations

Dataset coverage is limited. Evidence extraction can fail. Negatives are not yet
represented in classifier training input. Softmax thresholds are heuristic and
uncalibrated. Clinical note pretraining may not match patients' everyday language.
The system cannot confirm a diagnosis, prescribe medication, or book real care.
Mocked software tests verify behaviour, not medical safety or accuracy.

## Completed evaluation — 17 September 2026

Bio_ClinicalBERT achieved 98.76% complete-history and 85.04% partial-history validation accuracy. DistilBERT achieved 98.89% and 86.23%, respectively, and was retained. On 2,434 held-out test cases, DistilBERT achieved 99.26% complete-history accuracy (macro F1 99.25%) and 85.99% partial-history accuracy (macro F1 86.18%). These are synthetic-data results, not clinical accuracy.

Recorded OpenAI voice input now has a transcript review step: Send transcribes, then a second Send submits the reviewed text. Live OpenAI transcription is not implemented.
