"""Evaluate base vs QLoRA-tuned model on the held-out test split and write reports/metrics.json."""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data import build_messages, read_jsonl
from .metrics import score


def load(cfg, adapter=None):
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], quantization_config=bnb, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


@torch.no_grad()
def generate(model, tok, texts, batch_size=16, max_new_tokens=96):
    outs, start = [], time.time()
    for i in range(0, len(texts), batch_size):
        prompts = [tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
                   for t in texts[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs, (time.time() - start) / len(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora.yaml")
    ap.add_argument("--adapter", default=None, help="defaults to output_dir from config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    adapter = args.adapter or cfg["output_dir"]
    test = read_jsonl(Path(cfg["data_dir"]) / "test.jsonl")
    texts, golds = [r["text"] for r in test], [r["target"] for r in test]

    report = {"model": cfg["model_name"], "n_test": len(test)}
    for name, ad in [("base", None), ("qlora", adapter)]:
        model, tok = load(cfg, ad)
        preds, sec = generate(model, tok, texts)
        report[name] = score(preds, golds) | {"sec_per_example": sec}
        print(name, json.dumps(report[name], indent=2))
        del model
        torch.cuda.empty_cache()
    Path("reports").mkdir(exist_ok=True)
    json.dump(report, open("reports/metrics.json", "w"), indent=2)
    print("Wrote reports/metrics.json")


if __name__ == "__main__":
    main()
