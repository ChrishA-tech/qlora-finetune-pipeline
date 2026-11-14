"""Evaluation metrics: JSON validity, per-field accuracy and micro F1 over (field, value) pairs."""
import json
import re

FIELDS = ["line_item", "amount", "tax_year", "category"]


def parse_json(text: str):
    """Return the first JSON object found in text, or None if absent or invalid."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _same(field: str, a, b) -> bool:
    if field == "amount":
        try:
            return abs(float(a) - float(b)) < 0.005
        except (TypeError, ValueError):
            return False
    return a == b


def score(predictions: list, targets: list) -> dict:
    """predictions: raw model strings; targets: gold dicts. Micro F1 over (field, value) pairs."""
    assert len(predictions) == len(targets)
    valid = tp = fp = fn = 0
    field_hits = {f: 0 for f in FIELDS}
    exact = 0
    for raw, gold in zip(predictions, targets):
        pred = parse_json(raw)
        if pred is None:
            fn += len(FIELDS)
            continue
        valid += 1
        hits = 0
        for f in FIELDS:
            if f in pred and _same(f, pred[f], gold[f]):
                tp += 1
                hits += 1
                field_hits[f] += 1
            else:
                fn += 1
                if f in pred:
                    fp += 1
        exact += hits == len(FIELDS)
    n = len(targets)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "json_valid_rate": valid / n,
        "exact_match": exact / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "field_accuracy": {f: field_hits[f] / n for f in FIELDS},
    }
