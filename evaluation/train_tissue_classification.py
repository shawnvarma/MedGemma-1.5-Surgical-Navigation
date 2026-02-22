"""
MedGemma 1.5 4B Fine-Tune for Binary Tissue Classification (TUMOR vs EDEMA)

Trains a LoRA adapter to identify tumor tissue type at a marked position
in FLAIR MRI slices.

CLASSES:
- TUMOR: Combined NCR_NET + ENHANCING (resection targets)
- EDEMA: Peritumoral edema (functional tissue, caution needed)

Rationale: 3-class LoRA had 60% NCR_NET->ENHANCING confusion. Binary
classification eliminates intra-tumor confusion while preserving the
clinically critical TUMOR vs EDEMA distinction.

Input: FLAIR slice with crosshair marker at query position
Output: Single token tissue classification

Usage:
    # First generate training data
    python generate_tissue_data_binary.py

    # Then train
    conda activate medgemma_challenge
    python train_tissue_classification_binary.py

    # Or with options
    python train_tissue_classification_binary.py --epochs 2 --use_4bit

    # Resume from latest checkpoint
    python train_tissue_classification_binary.py --resume latest --run_name <run_name>
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Binary tissue classes
# TUMOR = NCR_NET (label 1) + ENHANCING (label 4)
# EDEMA = Peritumoral edema (label 2)
TISSUE_CLASSES = ["TUMOR", "EDEMA"]

# Standard prompt for binary tissue classification
# Note: Says "tumor tissue" but LoRA is robust to wording — eval_prompt_variations.py
# confirmed 77-78% accuracy across "tumor tissue", "tissue", and "brain tissue" variants.
CLASSIFICATION_PROMPT = (
    "What type of brain tumor tissue is at the marked location in this MRI? "
    "Answer with exactly one word: TUMOR or EDEMA."
)


def draw_crosshair(image: Image.Image, x: int, y: int,
                   size: int = 10, color: str = 'red',
                   thickness: int = 3) -> Image.Image:
    """Draw a crosshair marker at (x, y) on the image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Horizontal line
    draw.line([(x - size, y), (x + size, y)], fill=color, width=thickness)
    # Vertical line
    draw.line([(x, y - size), (x, y + size)], fill=color, width=thickness)

    return img_copy


class TissueClassificationDataset(Dataset):
    """
    Dataset for binary tissue classification training.

    Expects JSON file with samples:
    [
        {
            "image_path": "path/to/slice.png",
            "x": 120,
            "y": 85,
            "label": "TUMOR"  # or "EDEMA"
        },
        ...
    ]
    """

    def __init__(self, data_path: str, crosshair_size: int = 10,
                 crosshair_color: str = 'red', crosshair_thickness: int = 3):
        with open(data_path) as f:
            self.samples = json.load(f)

        self.crosshair_size = crosshair_size
        self.crosshair_color = crosshair_color
        self.crosshair_thickness = crosshair_thickness

        # Validate labels
        for sample in self.samples:
            if sample['label'] not in TISSUE_CLASSES:
                raise ValueError(f"Invalid label: {sample['label']}. Expected one of {TISSUE_CLASSES}")

        print(f"  Loaded {len(self.samples)} samples")

        # Print class distribution
        label_counts = {}
        for sample in self.samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  Class distribution: {label_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Draw crosshair at the query position
        image = draw_crosshair(
            image,
            x=sample['x'],
            y=sample['y'],
            size=self.crosshair_size,
            color=self.crosshair_color,
            thickness=self.crosshair_thickness
        )

        # Build conversation format for SFTTrainer
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": CLASSIFICATION_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample['label']}
                ]
            }
        ]

        return {
            "messages": messages,
            "images": [image],
        }


def create_collate_fn(processor):
    """Create collate function that processes text and images."""

    def collate_fn(examples: list[dict[str, Any]]):
        texts = []
        images = []

        for example in examples:
            images.append(example["images"])

            # Apply chat template
            text = processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False
            ).strip()
            texts.append(text)

        # Tokenize and process images
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Create labels from input_ids
        labels = batch["input_ids"].clone()

        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask image tokens
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma 1.5 for binary tissue classification (TUMOR vs EDEMA)"
    )

    # Data paths (default to binary data)
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data JSON")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation data JSON")
    parser.add_argument("--output_dir", type=str, default="./tissue_runs",
                        help="Output directory for checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (default: timestamp)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch size)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")

    # LoRA hyperparameters
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")

    # Quantization
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA) - enabled by default")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")

    # Crosshair settings (should match inference)
    parser.add_argument("--crosshair_size", type=int, default=10,
                        help="Crosshair arm length in pixels")
    parser.add_argument("--crosshair_color", type=str, default="red",
                        help="Crosshair color")
    parser.add_argument("--crosshair_thickness", type=int, default=3,
                        help="Crosshair line thickness")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint. Pass 'latest' or a checkpoint path")

    args = parser.parse_args()

    # Handle 4-bit flag
    use_4bit = args.use_4bit and not args.no_4bit

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"tissue_binary_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MedGemma 1.5 Binary Tissue Classification Fine-tuning")
    print("TUMOR (NCR_NET + ENHANCING) vs EDEMA")
    print("=" * 60)
    print(f"\nOutput directory: {run_dir}")

    # Save config
    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["start_time"] = datetime.now().isoformat()
    config["tissue_classes"] = TISSUE_CLASSES
    config["prompt"] = CLASSIFICATION_PROMPT
    config["use_4bit"] = use_4bit
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check data exists
    if not Path(args.train_data).exists():
        print(f"\nERROR: Training data not found: {args.train_data}")
        print("Run generate_tissue_data_binary.py first to create training data.")
        return

    if not Path(args.val_data).exists():
        print(f"\nERROR: Validation data not found: {args.val_data}")
        print("Run generate_tissue_data_binary.py first to create training data.")
        return

    # Load model
    print("\nLoading MedGemma 1.5 4B-IT...")
    model_id = "google/medgemma-1.5-4b-it"

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if use_4bit:
        print("  Using 4-bit quantization (QLoRA)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    # Use right padding for training
    processor.tokenizer.padding_side = "right"

    print(f"  Model loaded. Parameters: {model.num_parameters():,}")

    # Configure LoRA
    print(f"\nConfiguring LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TissueClassificationDataset(
        args.train_data,
        crosshair_size=args.crosshair_size,
        crosshair_color=args.crosshair_color,
        crosshair_thickness=args.crosshair_thickness
    )
    val_dataset = TissueClassificationDataset(
        args.val_data,
        crosshair_size=args.crosshair_size,
        crosshair_color=args.crosshair_color,
        crosshair_thickness=args.crosshair_thickness
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Calculate steps
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"\n  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")

    # Training config
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

    # Create trainer
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
    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Crosshair: size={args.crosshair_size}, color={args.crosshair_color}")
    print(f"  Classes: {TISSUE_CLASSES}")
    print("-" * 60)

    # Handle resume from checkpoint
    resume_from_checkpoint = None
    if args.resume:
        if args.resume.lower() == "latest":
            # Find latest checkpoint in run_dir
            checkpoints = list(run_dir.glob("checkpoint-*"))
            if checkpoints:
                resume_from_checkpoint = str(max(checkpoints, key=lambda p: int(p.name.split("-")[1])))
                print(f"  Resuming from: {resume_from_checkpoint}")
            else:
                print("  WARNING: No checkpoints found, starting fresh")
        else:
            resume_from_checkpoint = args.resume
            print(f"  Resuming from: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print("\nSaving final model...")
    final_path = run_dir / "final_adapter"
    trainer.save_model(str(final_path))

    # Also save processor for easy loading
    processor.save_pretrained(str(final_path))

    # Save completion info
    completion_info = {
        "completed_at": datetime.now().isoformat(),
        "final_adapter": str(final_path),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": args.epochs,
        "tissue_classes": TISSUE_CLASSES,
        "rationale": "Binary classification (TUMOR vs EDEMA) to eliminate NCR_NET<->ENHANCING confusion from 3-class model."
    }
    with open(run_dir / "completion.json", "w") as f:
        json.dump(completion_info, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Checkpoints: {run_dir}")
    print(f"  Final adapter: {final_path}")
    print("\nTo use the adapter:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{final_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()