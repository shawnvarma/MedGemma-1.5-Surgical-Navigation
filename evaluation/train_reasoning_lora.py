"""
MedGemma 1.5 4B-IT LoRA Fine-tuning: Reasoning Distillation

Fine-tunes MedGemma on Gemini reasoning traces using LoRA.
Teaches MedGemma to produce surgical navigation reasoning from
GT-injected prompts (same inputs, better outputs).

Usage:
    conda activate medgemma

    # Quick trial (default: 250 train, 60 val from traces)
    python evaluation/train_reasoning_lora.py --traces_file /path/to/traces/train.json

    # Custom split sizes
    python evaluation/train_reasoning_lora.py --traces_file ... --train_size 200 --val_size 50

    # Resume from checkpoint
    python evaluation/train_reasoning_lora.py --traces_file ... --resume_from experiments/checkpoint-100

Environment variables:
    TRACES_DIR   -- directory containing train.json (default: ./reasoning_traces)
    OUTPUT_DIR   -- where to write checkpoints (default: ./experiments)
"""

import os
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRACES_DIR = Path(os.environ.get("TRACES_DIR", "reasoning_traces"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments"))
MAX_IMAGES = 3  # VRAM limit on RTX 4080 16GB: top margin, peak tumor, bottom margin


def load_and_split_traces(traces_path: Path, train_size: int, val_size: int,
                          seed: int = 42) -> tuple[list, list, list]:
    """Load traces, filter to grounded, split into train/val/test."""
    with open(traces_path) as f:
        all_traces = json.load(f)

    # Filter to grounded only
    grounded = [t for t in all_traces if t.get("grounded", False)]
    ungrounded = len(all_traces) - len(grounded)
    print(f"  Loaded {len(all_traces)} traces, {len(grounded)} grounded ({ungrounded} filtered out)")

    random.seed(seed)
    random.shuffle(grounded)

    # Split
    train = grounded[:train_size]
    val = grounded[train_size:train_size + val_size]
    test = grounded[train_size + val_size:]

    return train, val, test


def select_3_slices(trace: dict) -> list[int]:
    """Pick 3 representative slice indices from the 5-slice corridor.
    Returns indices into the images/slice_indices arrays (not the slice numbers).
    Strategy: first (top margin), middle (peak tumor), last (bottom margin)."""
    n = len(trace["images"])
    if n <= 3:
        return list(range(n))
    return [0, n // 2, n - 1]


def rebuild_user_content(trace: dict, selected: list[int]) -> list[dict]:
    """Rebuild the user content for the selected subset of slices."""
    original_content = trace["conversations"][1]["content"]

    # Extract per-slice parts (image + text pairs) and the final prompt
    # Original structure: [image, text, image, text, ..., image, text, prompt_text]
    n_slices = len(trace["images"])
    parts_per_slice = []
    for i in range(n_slices):
        img_part = original_content[i * 2]      # {"type": "image", "image_index": i}
        txt_part = original_content[i * 2 + 1]  # {"type": "text", "text": "SLICE ..."}
        parts_per_slice.append((img_part, txt_part))

    # The prompt is the last element
    prompt_part = original_content[-1]

    # Build new content with only selected slices
    new_content = []
    for new_idx, orig_idx in enumerate(selected):
        new_content.append({"type": "image"})  # image placeholder
        new_content.append(parts_per_slice[orig_idx][1])  # slice text label
    new_content.append(prompt_part)  # verified measurements + instructions

    return new_content


class ReasoningTraceDataset(Dataset):
    """Dataset for MedGemma reasoning distillation training."""

    def __init__(self, traces: list):
        self.traces = traces
        print(f"    Dataset: {len(traces)} samples")

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]

        # Select 3 slices from the 5-slice corridor
        selected = select_3_slices(trace)

        # Load images for selected slices
        images = []
        for i in selected:
            img = Image.open(trace["images"][i]).convert("RGB")
            images.append(img)

        # Rebuild user content for selected slices
        user_content = rebuild_user_content(trace, selected)

        # Get the Gemini reasoning trace as the target
        assistant_text = trace["conversations"][2]["content"]

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [
                {"type": "text", "text": assistant_text}
            ]}
        ]

        return {
            "messages": messages,
            "images": images,
        }


def create_collate_fn(processor):
    """Create collate function that processes text and images."""

    def collate_fn(examples: list[dict[str, Any]]):
        texts = []
        images = []

        for example in examples:
            images.append(example["images"])

            text = processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False
            ).strip()
            texts.append(text)

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Create labels: mask everything except the assistant's response
        labels = batch["input_ids"].clone()

        # Mask padding
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask image tokens
        if hasattr(processor.tokenizer, "special_tokens_map") and "boi_token" in processor.tokenizer.special_tokens_map:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
            labels[labels == image_token_id] = -100

        # Mask Gemma image placeholder token
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma on reasoning traces")
    parser.add_argument("--traces_file", type=str,
                        default=str(TRACES_DIR / "train.json"),
                        help="Path to traces JSON (will be split internally)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--run_name", type=str, default=None)

    # Split sizes
    parser.add_argument("--train_size", type=int, default=250)
    parser.add_argument("--val_size", type=int, default=60)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--logging_steps", type=int, default=5)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"reasoning_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MedGemma 1.5 LoRA Fine-tuning -- Reasoning Distillation")
    print("=" * 60)
    print(f"Output: {run_dir}")

    # Load and split traces
    print(f"\nLoading traces from {args.traces_file}...")
    train_traces, val_traces, test_traces = load_and_split_traces(
        Path(args.traces_file), args.train_size, args.val_size, args.seed
    )
    print(f"  Train: {len(train_traces)}, Val: {len(val_traces)}, Held-out: {len(test_traces)}")

    # Save the split for reproducibility
    split_info = {
        "train_patients": [t["patient_id"] for t in train_traces],
        "val_patients": [t["patient_id"] for t in val_traces],
        "test_patients": [t["patient_id"] for t in test_traces],
        "seed": args.seed,
    }
    with open(run_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Save config
    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["start_time"] = datetime.now().isoformat()
    config["max_images_per_sample"] = MAX_IMAGES
    config["train_samples"] = len(train_traces)
    config["val_samples"] = len(val_traces)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # GPU check
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print("\nLoading MedGemma 1.5 4B-IT...")
    model_id = "google/medgemma-1.5-4b-it"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        ),
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    print(f"  Parameters: {model.num_parameters():,}")

    # LoRA config
    print(f"\nLoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Datasets
    print("\nBuilding datasets...")
    train_dataset = ReasoningTraceDataset(train_traces)
    val_dataset = ReasoningTraceDataset(val_traces)

    # Training config
    steps_per_epoch = len(train_traces) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    print(f"\n  Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")

    sft_config = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        report_to=["tensorboard"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # Windows compatibility
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=create_collate_fn(processor),
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.lr}")
    print(f"  Images/sample: {MAX_IMAGES}")
    print("=" * 60)

    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final
    print("\nSaving final model...")
    final_path = run_dir / "final_checkpoint"
    trainer.save_model(str(final_path))

    completion = {
        "completed_at": datetime.now().isoformat(),
        "final_checkpoint": str(final_path),
        "train_samples": len(train_traces),
        "val_samples": len(val_traces),
        "test_samples": len(test_traces),
        "epochs": args.epochs,
        "max_images": MAX_IMAGES,
    }
    with open(run_dir / "completion.json", "w") as f:
        json.dump(completion, f, indent=2)

    print(f"\nDone!")
    print(f"  Checkpoint: {final_path}")
    print(f"  Next: python evaluation/eval_reasoning.py --checkpoint {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
