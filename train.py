"""
Gemma 3 1B - Supervised Fine-Tuning (SFT) on Stanford Alpaca
Runs inside a Vertex AI custom training container.
"""

import argparse
import os
import subprocess

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ALPACA_PROMPT = """\
Below is an instruction that describes a task{input_block}. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}
{input_section}
### Response:
{output}"""


def format_alpaca(example):
    has_input = bool(example.get("input", "").strip())
    text = ALPACA_PROMPT.format(
        input_block=", paired with an input that provides further context" if has_input else "",
        instruction=example["instruction"],
        input_section=f"\n### Input:\n{example['input']}\n" if has_input else "",
        output=example["output"],
    )
    return {"text": text}


def main(args):
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=0.05, seed=42)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        max_seq_length=512,
        dataset_text_field="text",
        logging_steps=25,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_32bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Vertex AI sets AIP_MODEL_DIR; upload there if set
    gcs_dest = args.gcs_output or os.environ.get("AIP_MODEL_DIR")
    if gcs_dest:
        print(f"Uploading model to {gcs_dest}")
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", args.output_dir + "/", gcs_dest],
            check=True,
        )
        print("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT fine-tune Gemma 3 1B on Alpaca")
    parser.add_argument("--model_name", default="google/gemma-3-1b-pt",
                        help="HuggingFace model ID")
    parser.add_argument("--output_dir", default="/tmp/gemma-alpaca-sft",
                        help="Local directory to save the trained model")
    parser.add_argument("--gcs_output", default=None,
                        help="GCS path to upload the final model (e.g. gs://bucket/model)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (required for gated Gemma model)")
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass --hf_token. "
            "Get yours at https://huggingface.co/settings/tokens"
        )

    main(args)
