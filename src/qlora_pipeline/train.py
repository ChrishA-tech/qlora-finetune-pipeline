"""QLoRA fine-tuning with Hugging Face Transformers + PEFT (4-bit NF4 base, LoRA adapters)."""
import argparse

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

from .data import build_messages, read_jsonl, target_to_str
from pathlib import Path


class ExtractionDataset(Dataset):
    """Tokenises prompt + JSON answer; loss is computed on the answer tokens only."""

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for r in rows:
            prompt = tokenizer.apply_chat_template(build_messages(r["text"]), tokenize=False, add_generation_prompt=True)
            answer = target_to_str(r["target"]) + tokenizer.eos_token
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            ids = (p_ids + a_ids)[:max_len]
            labels = ([-100] * len(p_ids) + a_ids)[:max_len]
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    set_seed(cfg["seed"])
    q, lo, tr = cfg["quantization"], cfg["lora"], cfg["training"]

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
        bnb_4bit_use_double_quant=q["double_quant"], bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
        target_modules=lo["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    data_dir = Path(cfg["data_dir"])
    train_ds = ExtractionDataset(read_jsonl(data_dir / "train.jsonl"), tok, cfg["max_seq_len"])
    val_ds = ExtractionDataset(read_jsonl(data_dir / "val.jsonl"), tok, cfg["max_seq_len"])

    args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=tr["epochs"],
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=tr["per_device_batch_size"],
        gradient_accumulation_steps=tr["grad_accum_steps"], learning_rate=tr["learning_rate"],
        warmup_ratio=tr["warmup_ratio"], logging_steps=tr["logging_steps"],
        eval_strategy="steps", eval_steps=tr["eval_steps"], save_strategy="steps",
        save_steps=tr["eval_steps"], save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(), optim="paged_adamw_8bit",
        lr_scheduler_type="cosine", report_to=tr["report_to"], seed=cfg["seed"])
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"Adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
