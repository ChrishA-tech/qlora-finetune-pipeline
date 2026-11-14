from qlora_pipeline.metrics import parse_json, score

GOLD = {"line_item": "Standard deduction", "amount": 1000.0, "tax_year": 2023, "category": "deduction"}


def test_parse_json_handles_wrapping_text():
    assert parse_json('Sure! {"a": 1} done')["a"] == 1
    assert parse_json("no json here") is None


def test_perfect_prediction():
    r = score(['{"line_item":"Standard deduction","amount":1000,"tax_year":2023,"category":"deduction"}'], [GOLD])
    assert r["f1"] == 1.0 and r["exact_match"] == 1.0 and r["json_valid_rate"] == 1.0


def test_invalid_json_counts_as_misses():
    r = score(["garbage"], [GOLD])
    assert r["json_valid_rate"] == 0.0 and r["f1"] == 0.0


def test_partial_prediction():
    r = score(['{"line_item":"Standard deduction","amount":999,"tax_year":2023,"category":"deduction"}'], [GOLD])
    assert r["exact_match"] == 0.0 and 0.0 < r["f1"] < 1.0
    assert r["field_accuracy"]["amount"] == 0.0
