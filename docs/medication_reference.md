# DDx reference preprocessing

This additive update preprocesses the user's nursing-friend medicine list. It does
not change the existing classifier, symptom preprocessing, or web application.

Run `python scripts/preprocess_medication_reference.py` from this folder or after copying
the files into the DDx project. No dependencies, downloads, API calls, or model
training are required. Run `python -m unittest discover -s tests -p test_medication_reference.py`
to check parsing and draft status.

Input: `data/reference/condition_medication_reference.txt`, a compact transcription of the
30-category user message. The broken Paracetamol line is joined, brands sharing
the same ingredients are grouped, and combinations remain separate from single
ingredients. Ingredient salts are not merged. Source associations are preserved,
including claims needing correction; none is medically approved by this import.

Output: `data/reference/medication_reference_draft.json`. Medicines are deduplicated across
conditions; each condition association retains its own supplied names and input
line. The source hash records the exact compact input used. Original references,
formulation, strength, route, indications, contraindications, and prescribing
eligibility have not been verified. Empty sources mean missing evidence.

All associations have `recommendation_enabled: false`. Do not feed this draft
into patient-facing answers, model training, or medication recommendations.
Reviewed reference content and runtime eligibility rules are separate future work.
This list contains no symptom examples and adds no classifier disease coverage.
