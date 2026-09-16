# Stage one: conversation and symptom handling

## Implemented

- Separate pending question from interview history. A yes/no response cannot
  accidentally answer an older question after an assessment.
- Accept explicit corrections, preserve negative answers, and remove an earlier
  finding when the user explicitly says it is unknown. Unknown is not No.
- Normalize whitespace, Unicode apostrophes, code casing, and boolean casing
  during extraction validation. Quotes must still match the user's message.
- Preserve accepted findings and expose extraction counts in the existing local
  developer endpoint. Empty extraction gets one clarification, then an explicit
  uncertainty response instead of repeating the same generic question.
- Handle requests for summaries, clarifications, medication information, and
  appointments separately from symptom evidence. The latter two clearly explain
  that their services are not connected in this stage.
- Continue discussions after an assessment or uncertainty result. Both backend
  and frontend have been updated. Urgent guidance remains prominent.
- Optional generated wording for interview questions. The local classifier alone
  supplies condition predictions. No diagnosis or medicine generation is requested
  from the question-rephrasing model.

## Cost and operation

Most substantive turns make one extraction request and optionally one short
question-rephrasing request. Simple pending yes/no answers use local parsing.
Duplicate successful submissions reuse their response. There are no automatic
paid retries. Existing cumulative request caps remain unchanged; they are not a
dollar spending guarantee. To turn off question rephrasing, add
`NATURAL_REPLIES=false` to backend/.env and restart. Never share that file.

Tests use fake API responses and fake classifier outputs, with no model downloads
or paid calls. Run `python -m unittest discover -s tests -p test_conversation.py -v`.
They test orchestration, not clinical accuracy or live extraction reliability.

## Files

- backend/app.py: endpoint orchestration and session lifecycle.
- backend/core.py: evidence validation and existing classifier input formatting.
- backend/conversation.py: interpretation schema and conversation decisions.
- backend/dialogue.py: optional bounded question rephrasing, with local fallback.
- frontend/src/main.tsx: permit continued input after non-urgent assessments.
- tests/test_conversation.py: offline regression checks.

## Limits and next stages

The trained model remains the existing DistilBERT checkpoint. Bio_ClinicalBERT
training, dataset expansion, continuous voice mode, appointments, and reviewed
medicine content are separate stages and are not implemented by this update.
The old frontend WhatsApp handoff remains until the appointment stage replaces it.

Malaria is absent from the classifier's 49 labels. Explicitly named unsupported
conditions trigger an uncertainty response; this is not a comprehensive
out-of-distribution detector. A disease the user does not name can still be outside
coverage. Correctly extracted symptoms do not make the classifier clinically valid.

Negative findings remain excluded from classifier input because its training data
did not represent them that way. Thresholds are uncalibrated heuristics. The
question rewriter is constrained by prompting, not clinically validated semantic
verification. Real conversation/voice checks and medical evaluation remain needed.

The original symptom-loop screenshot does not establish the exact extraction
failure. The update handles plausible formatting failures and removes the repeated
fallback path; live replay is still necessary to assess the original case.

Restart with `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`
from the project root. Keep the Vite frontend running in a second terminal.
Restarting clears in-memory sessions; begin a new conversation after installing.
