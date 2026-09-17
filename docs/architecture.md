# Architecture

## Request lifecycle

React creates a server session with age and recorded sex. Those fields reflect the
dataset's representation, which currently has only two sex categories. Each sent
message receives a UUID. Repeating a completed UUID returns its cached response
rather than making another paid request.

FastAPI resolves simple yes/no answers only against the currently pending question.
Other messages use structured extraction. The extractor receives the evidence
catalog, recent conversation, and known findings. It returns quoted observations
and an intent, not a diagnosis. The backend validates codes, allowed values and
source quotes. Explicit corrections update prior evidence; uncertainty removes it.

Accepted evidence is formatted like training data and passed to the local saved
classifier. A heuristic selects follow-ups or a preliminary result. The policy
handles urgent concerns, unsupported named conditions, clarification, summaries,
and appointment requests. Interview wording may be rewritten by a separate bounded
language-model call. Failed rephrasing falls back to the original question.

The frontend displays the reply. In voice mode it speaks automatically, then
reopens the microphone. Appointment actions open the browser-only booking section.

## File responsibilities

| File | Responsibility |
|---|---|
| backend/app.py | FastAPI endpoints, session locks, local model load, usage caps |
| backend/core.py | Evidence validation, classifier input, follow-up heuristic |
| backend/conversation.py | Intent schema, corrections, uncertainty and conversation decisions |
| backend/dialogue.py | Optional question rephrasing with fallback |
| frontend/src/main.tsx | App shell, chat transport and feature integration |
| frontend/src/hooks/voiceInput.ts | Browser recognition and recording adapters |
| frontend/src/hooks/useVoiceConversation.ts | Listen/send/speak/resume lifecycle |
| frontend/src/components/AppointmentBooking.tsx | Demo booking UI and persistence |
| frontend/src/data/demoPractitioners.ts | Fictional profiles and date/slot validation |
| scripts/classifier_data.py | Deterministic full/partial histories and evaluation metrics |
| scripts/classifier_training.py | Optimization, checkpoint save, validation and best-model selection |
| scripts/train_clinicalbert.py | Bio_ClinicalBERT experiment entry point |
| scripts/evaluate_classifier.py | Comparable saved-model evaluation |

## API endpoints

- `GET /health`: model status and local request counts, never an API key.
- `POST /sessions`: begin a session.
- `POST /sessions/{id}/messages`: submit a text/transcribed turn.
- `POST /sessions/{id}/transcribe`: transcribe a submitted recording.
- `POST /sessions/{id}/speech`: optional paid speech route retained, disabled by
  the example cap and not used by the current frontend.
- `DELETE /sessions/{id}`: clear a session.
- `GET /sessions/{id}/debug`: local developer evidence inspection. Contains session
  findings; do not expose or share it as a public service.

The API is bound to loopback and configured for local origins. It is not a
production multi-user medical service. A public deployment would need a separate
privacy/security/clinical design, not just a change of bind address.

## Storage

Conversation memory: temporary server RAM, cleared on restart. Usage counts:
backend/usage_counts.json. Model weights: ignored models directory. Demo bookings:
localStorage `ddx.demoAppointments.v1`, removable using Cancel. No appointment
request is sent to a clinician and no health history is stored with a booking.
