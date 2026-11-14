"""Synthetic line-item extraction dataset (no real or personal data).

Task: turn a messy financial/tax line item into schema-constrained JSON:
{"line_item": str, "amount": float, "tax_year": int, "category": str}
"""
import argparse
import json
import random
from pathlib import Path

import yaml

CATEGORIES = ["income", "deduction", "credit", "payment"]
LINE_ITEMS = {
    "income": ["Wages and salaries", "Taxable interest", "Ordinary dividends", "Business income", "Rental income"],
    "deduction": ["Standard deduction", "Mortgage interest", "Charitable contributions", "State and local taxes", "Student loan interest"],
    "credit": ["Child tax credit", "Education credit", "Foreign tax credit", "Energy efficiency credit"],
    "payment": ["Federal tax withheld", "Estimated tax payments", "Prior year overpayment applied"],
}
SYSTEM_PROMPT = (
    "Extract the line item from the text. Reply with only a JSON object with keys "
    '"line_item" (string), "amount" (number, negative if shown in brackets), '
    '"tax_year" (integer) and "category" (one of income, deduction, credit, payment).'
)


def _fmt_amount(amount: float, style: int, rng: random.Random) -> str:
    body = f"{abs(amount):,.2f}"
    neg = amount < 0
    if style == 0:
        return f"({body})" if neg else f"${body}"
    if style == 1:
        return f"{'-' if neg else ''}USD {abs(amount):.2f}"
    return f"{'-' if neg else ''}{body}"


def make_example(rng: random.Random) -> dict:
    category = rng.choice(CATEGORIES)
    item = rng.choice(LINE_ITEMS[category])
    year = rng.randint(2019, 2025)
    amount = round(rng.uniform(25, 150000), 2)
    if category in ("deduction", "payment") and rng.random() < 0.15:
        amount = -amount
    style = rng.randint(0, 2)
    amt = _fmt_amount(amount, style, rng)
    layout = rng.randint(0, 3)
    noise = rng.choice(["", " ", "  ", "*", " (see stmt)"])
    if layout == 0:
        text = f"{item}{noise}: {amt} [TY{year}]"
    elif layout == 1:
        text = f"TY {year} | {item.upper()} | {amt}"
    elif layout == 2:
        text = f"{item} ........ {amt}  (tax year {year})"
    else:
        text = f"{year}  {item}{noise}  {amt}"
    target = {"line_item": item, "amount": amount, "tax_year": year, "category": category}
    return {"text": text.strip(), "target": target}


def build_messages(text: str) -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]


def target_to_str(target: dict) -> str:
    return json.dumps(target, separators=(",", ":"))


def generate_splits(n_train: int, n_val: int, n_test: int, seed: int = 42) -> dict:
    """Generate unique examples (by text) and split with no overlap between splits."""
    rng = random.Random(seed)
    total = n_train + n_val + n_test
    seen, examples = set(), []
    while len(examples) < total:
        ex = make_example(rng)
        if ex["text"] in seen:
            continue
        seen.add(ex["text"])
        examples.append(ex)
    return {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qlora.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    splits = generate_splits(d["n_train"], d["n_val"], d["n_test"], cfg["seed"])
    for name, rows in splits.items():
        write_jsonl(Path(cfg["data_dir"]) / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} examples")


if __name__ == "__main__":
    main()
