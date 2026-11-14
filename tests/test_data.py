from qlora_pipeline.data import generate_splits, make_example, target_to_str
import json, random


def test_deterministic():
    a = generate_splits(20, 5, 5, seed=1)
    b = generate_splits(20, 5, 5, seed=1)
    assert a == b


def test_no_leakage_between_splits():
    s = generate_splits(200, 30, 30, seed=3)
    texts = [set(r["text"] for r in s[k]) for k in ("train", "val", "test")]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])


def test_target_schema():
    ex = make_example(random.Random(0))
    t = json.loads(target_to_str(ex["target"]))
    assert set(t) == {"line_item", "amount", "tax_year", "category"}
    assert t["category"] in {"income", "deduction", "credit", "payment"}
