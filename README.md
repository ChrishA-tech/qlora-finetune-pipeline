# QLoRA Fine-Tuning Pipeline: Open-Weight LLM for Schema-Constrained Extraction

End-to-end, reproducible pipeline that fine-tunes an open-weight LLM with **QLoRA** (4-bit NF4 base + LoRA adapters) using **Hugging Face Transformers + PEFT**, evaluates it against the untuned base model, and serves it behind a REST API in **Docker** that can run fully offline (secure / air-gapped deployments).

## Problem
Financial and tax line items arrive in inconsistent layouts (`TY 2024 | WAGES AND SALARIES | $41,272.52`, `Standard deduction ........ (1,000.00)`, ...). Downstream systems need strict JSON:

```json
{"line_item": "Wages and salaries", "amount": 41272.52, "tax_year": 2024, "category": "income"}
```

A small open-weight model run locally avoids sending sensitive documents to an external API. The dataset is **synthetic** (no real or personal data), generated deterministically from a seed.

## Pipeline
| Stage | File | What it does |
|---|---|---|
| Data | `src/qlora_pipeline/data.py` | Seeded synthetic generator, de-duplicated train/val/test splits (no leakage) |
| Train | `src/qlora_pipeline/train.py` | QLoRA (NF4, double quant, paged 8-bit AdamW), loss on answer tokens only |
| Evaluate | `src/qlora_pipeline/evaluate.py` | Base vs tuned: JSON-valid rate, exact match, per-field accuracy, micro F1, latency |
| Serve | `src/qlora_pipeline/serve.py` | FastAPI `/extract` + `/health`, offline-capable |
| Package | `Dockerfile` | CUDA runtime image, `HF_HUB_OFFLINE=1` |
| Config | `configs/qlora.yaml` | Model, LoRA rank, quantisation, hyper-parameters in one place |

## Quick start (GPU required for training, e.g. free Colab T4)
```bash
pip install -r requirements.txt
export PYTHONPATH=src
make test     # unit tests for data + metrics (CPU only)
make data     # generate data/train|val|test.jsonl
make train    # QLoRA fine-tune -> outputs/adapter
make eval     # base vs tuned -> reports/metrics.json
make serve    # REST API on :8000

curl -X POST localhost:8000/extract -H 'Content-Type: application/json' \
  -d '{"text": "TY 2023 | MORTGAGE INTEREST | 8,412.10"}'
```

Docker: `make docker`, then run with `--gpus all -p 8000:8000 qlora-extractor`. For air-gapped hosts, copy the base model into the image or mount it, so nothing is downloaded at runtime.

Optional experiment tracking: set `training.report_to: mlflow` in the config.

## Results
> Fill this table in after running `make eval` on your own GPU. Do not publish numbers you have not reproduced.

| Model | JSON valid | Exact match | Micro F1 | Sec / example |
|---|---|---|---|---|
| Base (untuned) | | | | |
| QLoRA-tuned | | | | |

## Design choices
- **QLoRA over full fine-tuning:** a 1.5B model trains on a single 16 GB GPU; only the adapter (a few MB) is stored and versioned.
- **Loss on answer tokens only:** the model learns the output format, not to repeat the prompt.
- **Evaluation-led:** a held-out test split and field-level metrics, with the base model as the baseline.
- **Offline-first serving:** no external API calls at inference time.

## Limitations
Synthetic data is cleaner than real documents; the next step would be to test on real, consented samples and add a confidence threshold with human review for low-confidence outputs.
