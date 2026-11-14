"""FastAPI model-serving endpoint. Fully offline when HF_HUB_OFFLINE=1 and weights are local."""
import os

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .evaluate import generate, load
from .metrics import parse_json

CONFIG = os.environ.get("CONFIG_PATH", "configs/qlora.yaml")
ADAPTER = os.environ.get("ADAPTER_PATH", "outputs/adapter")
cfg = yaml.safe_load(open(CONFIG))
model, tok = load(cfg, ADAPTER)
app = FastAPI(title="Line-item extraction (QLoRA)")


class Req(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model": cfg["model_name"], "adapter": ADAPTER, "cuda": torch.cuda.is_available()}


@app.post("/extract")
def extract(req: Req):
    raw, _ = generate(model, tok, [req.text], batch_size=1)
    parsed = parse_json(raw[0])
    if parsed is None:
        raise HTTPException(status_code=422, detail={"error": "model output was not valid JSON", "raw": raw[0]})
    return parsed
