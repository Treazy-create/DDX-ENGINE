"""Build draft reference records; never modify classifier data or approve medicines."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def preprocess(text):
    conditions, medicines, associations = {}, {}, {}
    current = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            current = line[1:-1].strip()
            if not current or current.casefold() in conditions:
                raise ValueError(f'Line {number}: empty or duplicate condition')
            conditions[current.casefold()] = current
            continue
        if current is None or line.count('|') != 1:
            raise ValueError(f'Line {number}: expected ingredients | names')
        ingredients, names = line.split('|')
        ingredients = [v.strip() for v in ingredients.split('+')]
        names = [v.strip() for v in names.split(';')]
        if not all(ingredients + names):
            raise ValueError(f'Line {number}: empty ingredient or name')
        key = tuple(sorted(v.casefold() for v in ingredients))
        medicine_id = hashlib.sha256('|'.join(key).encode()).hexdigest()[:16]
        medicine = medicines.setdefault(medicine_id, {
            'id': medicine_id, 'ingredients': ingredients, 'supplied_names': [],
            'review_status': 'unverified', 'sources': [],
        })
        for name in names:
            if name.casefold() not in [v.casefold() for v in medicine['supplied_names']]:
                medicine['supplied_names'].append(name)
        association_key = (current.casefold(), medicine_id)
        if association_key in associations:
            raise ValueError(f'Line {number}: duplicate condition/medicine association')
        associations[association_key] = {
            'condition': current, 'medicine_id': medicine_id,
            'source_line': number, 'supplied_names': names,
            'review_status': 'unverified', 'recommendation_enabled': False,
        }
    if not conditions or not associations:
        raise ValueError('No reference records found')
    return {
        'schema_version': 1,
        'provenance': 'User-provided nursing-friend list; compact transcription',
        'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'purpose': 'Draft reference only; not diagnostic training or prescribing rules',
        'conditions': list(conditions.values()),
        'medicines': list(medicines.values()),
        'associations': list(associations.values()),
    }


def main():
    folder = ROOT / 'data' / 'reference'
    source = folder / 'condition_medication_reference.txt'
    result = preprocess(source.read_text(encoding='utf-8-sig'))
    destination = folder / 'medication_reference_draft.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Conditions: {len(result['conditions'])}")
    print(f"Unique ingredient combinations: {len(result['medicines'])}")
    print(f"Condition/medicine associations: {len(result['associations'])}")
    print('Automatic recommendations enabled: 0 (review required)')
    print(f'Saved: {destination}')


if __name__ == '__main__':
    main()
