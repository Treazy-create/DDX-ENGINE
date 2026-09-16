"""Dataset-aligned input formatting. No API calls, training, or diagnosis rules."""
import unicodedata


def normalized_text(value):
    value = unicodedata.normalize('NFKC', value)
    return ' '.join(value.translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'})).casefold().split())


def quote_matches(quote, message):
    return bool(normalized_text(quote)) and normalized_text(quote) in normalized_text(message)


def normalize_observations(items, definitions, message):
    accepted = {}
    for item in items:
        code, values, quote = item.code.strip().upper(), item.values, item.quote.strip()
        if code not in definitions or not quote_matches(quote, message):
            continue
        meta = definitions[code]
        if meta['data_type'] == 'B':
            values = [value.strip().capitalize() for value in values]
            if values in (['Yes'], ['No']):
                accepted[code] = values
        else:
            allowed = {str(value) for value in meta.get('possible-values', [])}
            if values and all(value in allowed for value in values):
                if meta['data_type'] == 'C' and len(values) != 1:
                    continue
                accepted[code] = list(dict.fromkeys(values))
    return accepted


def evidence_answer(meta, values):
    meaning = meta.get('value_meaning', {})
    result = []
    for value in values:
        decoded = meaning.get(value, value)
        if isinstance(decoded, dict):
            decoded = decoded.get('en', value)
        result.append(str(decoded))
    return '; '.join(result)


def model_evidence(answers, definitions):
    # Training had present evidence, not explicit negative binary findings.
    # Keep negatives in conversation state but do not invent unseen model input.
    return {
        code: values for code, values in answers.items()
        if not (definitions[code]['data_type'] == 'B' and values == ['No'])
    }


def model_text(age, sex, answers, definitions, initial):
    positive = model_evidence(answers, definitions)
    codes = sorted(positive, key=lambda code: (code != initial, code))
    lines = [f'Age: {age} years.', f"Recorded sex: {'male' if sex == 'M' else 'female'}."]
    for code in codes:
        meta = definitions[code]
        lines.append(f"{meta['question_en']} Answer: {evidence_answer(meta, positive[code])}.")
    return '\n'.join(lines)


def next_question(answers, asked, definitions, conditions, ranking):
    # Prefer questions associated with plausible model candidates. This is a
    # heuristic, not a clinically validated interviewing policy.
    choices = []
    for code, meta in definitions.items():
        if code in answers or code in asked:
            continue
        parent = meta.get('code_question', code)
        if parent != code and parent not in model_evidence(answers, definitions):
            continue
        weight = 0.0
        for label, probability in ranking:
            condition = conditions.get(label, {})
            if code in condition.get('symptoms', {}) or code in condition.get('antecedents', {}):
                weight += probability
        if weight > 0:
            # Favor evidence which distinguishes the leading candidates over
            # questions shared by every candidate. Metadata membership is a
            # rough proxy, not a measured information-gain estimate.
            total = sum(probability for _, probability in ranking) or 1.0
            fraction = weight / total
            score = fraction * (1 - fraction)
            if meta.get('is_antecedent'):
                score *= 0.7
            choices.append((score, code))
    return max(choices)[1] if choices else None


def short_answer(text, last_code, definitions):
    """Resolve simple replies locally, only against the question actually asked."""
    if last_code not in definitions or definitions[last_code]['data_type'] != 'B':
        return None
    answer = normalized_text(text).rstrip('.! ')
    if answer == 'both' and last_code == 'E_39':
        return {last_code: ['Yes']}
    if answer in ('yes', 'yes i do', 'yes, i do', 'yeah', 'yep'):
        return {last_code: ['Yes']}
    if answer in ('no', 'no i do not', 'no, i do not', 'nope', "no i don't", "no, i don't"):
        return {last_code: ['No']}
    if answer in ("i don't know", 'not sure', 'unsure', 'unknown'):
        return {}
    return None


def assessment_ready(answers, definitions, ranking, previous_top, followups):
    """Demo stopping heuristic; not calibrated clinical confidence."""
    positives = model_evidence(answers, definitions)
    symptoms = sum(not definitions[code].get('is_antecedent', False) for code in positives)
    if not ranking or symptoms < 3:
        return False
    lead, score = ranking[0]
    runner_up = ranking[1][1] if len(ranking) > 1 else 0.0
    stable = previous_top == lead or (followups == 0 and symptoms >= 5)
    return stable and score >= 0.75 and score - runner_up >= 0.20
