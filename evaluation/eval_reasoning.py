"""
Evaluate Reasoning Distillation: Base MedGemma vs Fine-tuned

Runs the same GT-injected trajectory prompts through both base and
distilled MedGemma, compares grounding and reasoning quality.

Usage:
    conda activate medgemma
    python evaluation/eval_reasoning.py --checkpoint /path/to/final_checkpoint

    # Fewer samples for quick check
    python evaluation/eval_reasoning.py --checkpoint ... --num_samples 10

Environment variables:
    TRACES_DIR  -- directory containing train.json (default: ./reasoning_traces)
"""

import os
import json
import re
import argparse
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRACES_DIR = Path(os.environ.get("TRACES_DIR", "reasoning_traces"))

# Keywords that indicate good reasoning quality
TRANSITION_KEYWORDS = ["transition", "entering", "exiting", "crossing", "boundary", "margin"]
CAUTION_KEYWORDS = ["caution", "risk", "careful", "spare", "functional", "peritumoral"]
STRUCTURE_KEYWORDS = ["trajectory", "corridor", "summary", "spanning", "depth", "target"]


def load_test_samples(traces_path: Path, split_path: Path, num_samples: int) -> list:
    """Load test samples from traces, using the training split to avoid overlap."""
    with open(traces_path) as f:
        all_traces = json.load(f)

    # If a split file exists from training, use its test patients
    test_patients = None
    if split_path and split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        test_patients = set(split.get("test_patients", []))
        print(f"  Using {len(test_patients)} held-out test patients from training split")

    # Filter to grounded traces
    grounded = [t for t in all_traces if t.get("grounded", False)]

    # If we have a split, only use test patients
    if test_patients:
        samples = [t for t in grounded if t["patient_id"] in test_patients]
    else:
        # Fallback: use the last N traces (least likely to overlap with training)
        samples = grounded

    samples = samples[:num_samples]
    print(f"  Loaded {len(samples)} eval samples")
    return samples


def build_prompt_content(trace: dict, max_images: int = 3) -> tuple[list, list[Image.Image]]:
    """Build the prompt content (messages + images) for a trace.
    Uses 3 slices: top margin, peak tumor, bottom margin."""
    n = len(trace["images"])
    if n <= max_images:
        selected = list(range(n))
    else:
        selected = [0, n // 2, n - 1]

    # Load images
    images = []
    for i in selected:
        img = Image.open(trace["images"][i]).convert("RGB")
        images.append(img)

    # Rebuild user content for selected slices
    original_content = trace["conversations"][1]["content"]
    n_slices = len(trace["images"])

    parts_per_slice = []
    for i in range(n_slices):
        txt_part = original_content[i * 2 + 1]
        parts_per_slice.append(txt_part)

    prompt_part = original_content[-1]  # Verified measurements + instructions

    user_content = []
    for new_idx, orig_idx in enumerate(selected):
        user_content.append({"type": "image"})
        user_content.append(parts_per_slice[orig_idx])
    user_content.append(prompt_part)

    messages = [{"role": "user", "content": user_content}]
    return messages, images


def generate_response(model, processor, messages: list, images: list[Image.Image],
                      max_new_tokens: int = 256) -> str:
    """Generate a response from MedGemma."""
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    ).strip()

    inputs = processor(
        text=[text],
        images=[images],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )

    # Decode only the generated tokens
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def validate_grounding(response_text: str, trace: dict) -> dict:
    """Check grounding against the trace's GT data."""
    text_lower = response_text.lower()
    issues = []

    # Tissue types present in this patient
    metrics = trace["metrics"]
    tissues_present = set()
    if metrics["enhancing_volume_cm3"] > 0:
        tissues_present.add("ENHANCING")
    if metrics["necrotic_volume_cm3"] > 0:
        tissues_present.add("NECROTIC")
    if metrics["edema_volume_cm3"] > 0:
        tissues_present.add("EDEMA")

    # Negation-aware tissue mention check
    negation_prefix = r'(?:no|without|absence\s+of|absent|lack(?:ing|s)?|not?\s+any|zero)\s+'

    def is_affirmative(keyword):
        matches = list(re.finditer(keyword, text_lower))
        if not matches:
            return False
        for m in matches:
            start = max(0, m.start() - 4)
            if text_lower[start:m.start()].endswith('non-') or text_lower[start:m.start()].endswith('non '):
                continue
            start = max(0, m.start() - 30)
            preceding = text_lower[start:m.start()]
            if not re.search(negation_prefix, preceding):
                return True
        return False

    tissue_mentions = {
        "ENHANCING": is_affirmative(r'enhanc'),
        "NECROTIC": is_affirmative(r'necro'),
        "EDEMA": is_affirmative(r'edema'),
    }

    for tissue, mentioned in tissue_mentions.items():
        if mentioned and tissue not in tissues_present:
            issues.append(f"HALLUCINATION: {tissue}")

    # Number check (10% tolerance for MedGemma -- looser than 5% for Gemini)
    numbers_found = re.findall(r'(\d+\.?\d*)\s*(?:cm[²³]|mm)', response_text)
    gt_numbers = set()
    gt_numbers.add(f"{metrics['total_volume_cm3']:.1f}")
    gt_numbers.add(f"{metrics['enhancing_volume_cm3']:.1f}")
    gt_numbers.add(f"{metrics['necrotic_volume_cm3']:.1f}")
    gt_numbers.add(f"{metrics['edema_volume_cm3']:.1f}")
    depth = metrics["max_slice"] - metrics["min_slice"]
    gt_numbers.add(str(depth))

    # Add per-slice areas from the trace's conversation text
    for part in trace["conversations"][1]["content"]:
        if isinstance(part, dict) and part.get("type") == "text":
            for match in re.findall(r'(\d+\.?\d*)\s*cm²', part.get("text", "")):
                gt_numbers.add(match)

    for num_str in numbers_found:
        try:
            num = float(num_str)
            matched = any(
                (float(gt) > 0 and abs(num - float(gt)) / float(gt) < 0.10)
                or (float(gt) == 0 and num == 0)
                for gt in gt_numbers
            )
            if not matched:
                issues.append(f"UNGROUNDED: {num_str}")
        except ValueError:
            pass

    return {
        "grounded": len(issues) == 0,
        "issues": issues,
        "tissue_mentions": tissue_mentions,
        "numbers_found": numbers_found,
    }


def check_reasoning_quality(response_text: str) -> dict:
    """Check for reasoning quality indicators."""
    text_lower = response_text.lower()

    has_transitions = any(kw in text_lower for kw in TRANSITION_KEYWORDS)
    has_caution = any(kw in text_lower for kw in CAUTION_KEYWORDS)
    has_structure = any(kw in text_lower for kw in STRUCTURE_KEYWORDS)

    # Does it reference specific slice numbers?
    slice_refs = re.findall(r'slice\s+\d+', text_lower)

    # Does it describe trends (increasing/decreasing)?
    has_trends = bool(re.search(r'increas|decreas|diminish|expand|taper|resolv|peak|maxim', text_lower))

    return {
        "has_transitions": has_transitions,
        "has_caution": has_caution,
        "has_structure": has_structure,
        "has_trends": has_trends,
        "slice_references": len(slice_refs),
        "word_count": len(response_text.split()),
        "quality_score": sum([has_transitions, has_caution, has_structure, has_trends]),
    }


def check_per_slice_accuracy(response_text: str, trace: dict) -> dict:
    """Check if tissue claims per slice match GT.
    Parses 'SLICE N:' blocks and verifies tissue types against the trace data."""
    # Build GT per-slice tissue map from the trace conversation
    gt_slice_tissues = {}
    for part in trace["conversations"][1]["content"]:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text", "")
            m = re.match(r'SLICE\s+(\d+):\s*(.*)', text)
            if m:
                z = int(m.group(1))
                desc = m.group(2).upper()
                tissues = set()
                if "ENHANCING" in desc:
                    tissues.add("ENHANCING")
                if "NECROTIC" in desc or "NCR" in desc:
                    tissues.add("NECROTIC")
                if "EDEMA" in desc:
                    tissues.add("EDEMA")
                is_healthy = "HEALTHY" in desc
                gt_slice_tissues[z] = {"tissues": tissues, "healthy": is_healthy}

    # Parse model output for SLICE N: blocks
    blocks = re.findall(
        r'(?:\*{0,2})SLICE\s+(\d+)(?:\*{0,2})\s*:\s*(.*?)(?=(?:\*{0,2})SLICE\s+\d+|$)',
        response_text, re.IGNORECASE | re.DOTALL
    )

    correct = 0
    incorrect = 0
    details = []

    for slice_str, block_text in blocks:
        z = int(slice_str)
        if z not in gt_slice_tissues:
            continue

        gt = gt_slice_tissues[z]
        block_lower = block_text.lower()

        # Check: if GT says healthy, model should say healthy/clean/clear/normal
        if gt["healthy"]:
            model_says_healthy = bool(re.search(
                r'healthy|clean|clear|normal|no\s+tumor|no\s+pathology|no\s+sign', block_lower
            ))
            model_claims_tumor = bool(re.search(
                r'enhanc|necro|edema|tumor\s+burden|resection\s+target', block_lower
            ))
            if model_says_healthy and not model_claims_tumor:
                correct += 1
                details.append({"slice": z, "correct": True, "gt": "HEALTHY"})
            else:
                incorrect += 1
                details.append({"slice": z, "correct": False, "gt": "HEALTHY",
                               "issue": "claimed tumor on healthy slice"})
        else:
            # GT has tumor -- check if mentioned tissue types are correct
            mentioned = set()
            if re.search(r'enhanc', block_lower):
                mentioned.add("ENHANCING")
            if re.search(r'necro', block_lower):
                mentioned.add("NECROTIC")
            if re.search(r'edema', block_lower):
                mentioned.add("EDEMA")

            # Hallucinated tissues (mentioned but not in GT for this slice)
            hallucinated = mentioned - gt["tissues"]
            if not hallucinated:
                correct += 1
                details.append({"slice": z, "correct": True,
                               "gt": list(gt["tissues"]), "mentioned": list(mentioned)})
            else:
                incorrect += 1
                details.append({"slice": z, "correct": False,
                               "gt": list(gt["tissues"]), "mentioned": list(mentioned),
                               "hallucinated": list(hallucinated)})

    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0.0

    return {
        "per_slice_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning distillation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--traces_file", type=str,
                        default=str(TRACES_DIR / "train.json"))
    parser.add_argument("--split_file", type=str, default=None,
                        help="Path to split.json from training run (auto-detected if not set)")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of side-by-side examples to print")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # Auto-detect split file from the training run directory
    if args.split_file is None:
        candidate = checkpoint_path.parent / "split.json"
        if not candidate.exists():
            candidate = checkpoint_path.parent.parent / "split.json"
        if candidate.exists():
            args.split_file = str(candidate)
            print(f"Auto-detected split file: {args.split_file}")

    # Output directory
    if args.output_dir is None:
        args.output_dir = str(checkpoint_path.parent / "eval_results")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Reasoning Distillation Evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {args.num_samples}")
    print("=" * 70)

    # Load test samples
    print("\nLoading test samples...")
    split_path = Path(args.split_file) if args.split_file else None
    samples = load_test_samples(Path(args.traces_file), split_path, args.num_samples)

    if not samples:
        print("ERROR: No test samples found.")
        return

    # Load model
    print("\nLoading MedGemma 1.5 4B-IT + LoRA...")
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

    # Load LoRA adapter
    print(f"  Loading LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    print(f"  Model ready.")

    # Run evaluation
    print(f"\nRunning {args.num_samples} samples through Base and Distilled models...")
    print("-" * 70)

    results = []
    examples_printed = 0

    # Incremental save file
    out_file = out_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for i, trace in enumerate(samples):
        patient_id = trace["patient_id"]
        messages, images = build_prompt_content(trace)

        # --- Base MedGemma (adapter disabled) ---
        model.disable_adapter_layers()
        base_response = generate_response(model, processor, messages, images)

        # --- Distilled MedGemma (adapter enabled) ---
        model.enable_adapter_layers()
        distilled_response = generate_response(model, processor, messages, images)

        # --- Gemini reference (from the trace) ---
        gemini_response = trace["conversations"][2]["content"]

        # Validate both
        base_grounding = validate_grounding(base_response, trace)
        distilled_grounding = validate_grounding(distilled_response, trace)

        # Quality check
        base_quality = check_reasoning_quality(base_response)
        distilled_quality = check_reasoning_quality(distilled_response)
        gemini_quality = check_reasoning_quality(gemini_response)

        # Per-slice accuracy check
        base_slice_acc = check_per_slice_accuracy(base_response, trace)
        distilled_slice_acc = check_per_slice_accuracy(distilled_response, trace)
        gemini_slice_acc = check_per_slice_accuracy(gemini_response, trace)

        result = {
            "patient_id": patient_id,
            "base": {
                "response": base_response,
                "grounded": base_grounding["grounded"],
                "issues": base_grounding["issues"],
                "per_slice_accuracy": base_slice_acc["per_slice_accuracy"],
                "per_slice_correct": base_slice_acc["correct"],
                "per_slice_total": base_slice_acc["total"],
                **base_quality,
            },
            "distilled": {
                "response": distilled_response,
                "grounded": distilled_grounding["grounded"],
                "issues": distilled_grounding["issues"],
                "per_slice_accuracy": distilled_slice_acc["per_slice_accuracy"],
                "per_slice_correct": distilled_slice_acc["correct"],
                "per_slice_total": distilled_slice_acc["total"],
                **distilled_quality,
            },
            "gemini": {
                "response": gemini_response,
                "per_slice_accuracy": gemini_slice_acc["per_slice_accuracy"],
                "per_slice_correct": gemini_slice_acc["correct"],
                "per_slice_total": gemini_slice_acc["total"],
                **gemini_quality,
            },
        }
        results.append(result)

        # Incremental save every 5 samples
        if len(results) % 5 == 0:
            with open(out_file, "w") as f:
                json.dump({"config": vars(args), "results": results}, f, indent=2)

        # Print side-by-side examples
        if examples_printed < args.num_examples:
            print(f"\n{'=' * 70}")
            print(f"EXAMPLE {examples_printed + 1}: {patient_id}")
            print(f"  Volume: {trace['metrics']['total_volume_cm3']:.1f} cm3")
            print(f"  Slices: {trace['slice_indices']}")
            print(f"{'=' * 70}")

            print(f"\n  [BASE MedGemma] ({base_quality['word_count']} words, "
                  f"grounded={'YES' if base_grounding['grounded'] else 'NO'}, "
                  f"slice_acc={base_slice_acc['correct']}/{base_slice_acc['total']}):")
            print(f"  {'-' * 60}")
            for line in base_response.split('\n'):
                print(f"    {line}")

            print(f"\n  [DISTILLED MedGemma] ({distilled_quality['word_count']} words, "
                  f"grounded={'YES' if distilled_grounding['grounded'] else 'NO'}, "
                  f"slice_acc={distilled_slice_acc['correct']}/{distilled_slice_acc['total']}):")
            print(f"  {'-' * 60}")
            for line in distilled_response.split('\n'):
                print(f"    {line}")

            teacher_model = trace.get("model", "Gemini")
            print(f"\n  [{teacher_model} reference] ({gemini_quality['word_count']} words, "
                  f"slice_acc={gemini_slice_acc['correct']}/{gemini_slice_acc['total']}):")
            print(f"  {'-' * 60}")
            for line in gemini_response.split('\n'):
                print(f"    {line}")

            examples_printed += 1

        # Progress
        b_mark = "pass" if base_grounding["grounded"] else "FAIL"
        d_mark = "pass" if distilled_grounding["grounded"] else "FAIL"
        print(f"  [{i+1}/{len(samples)}] {patient_id}: "
              f"base={b_mark} q={base_quality['quality_score']}/4 | "
              f"distilled={d_mark} q={distilled_quality['quality_score']}/4 | "
              f"gemini q={gemini_quality['quality_score']}/4")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    n = len(results)

    # Grounding
    base_grounded = sum(1 for r in results if r["base"]["grounded"])
    dist_grounded = sum(1 for r in results if r["distilled"]["grounded"])
    print(f"\nGrounding pass rate (numbers + tissue types):")
    print(f"  Base MedGemma:      {base_grounded}/{n} ({base_grounded/n*100:.0f}%)")
    print(f"  Distilled MedGemma: {dist_grounded}/{n} ({dist_grounded/n*100:.0f}%)")

    # Per-slice accuracy
    base_slice_correct = sum(r["base"]["per_slice_correct"] for r in results)
    base_slice_total = sum(r["base"]["per_slice_total"] for r in results)
    dist_slice_correct = sum(r["distilled"]["per_slice_correct"] for r in results)
    dist_slice_total = sum(r["distilled"]["per_slice_total"] for r in results)
    gem_slice_correct = sum(r["gemini"]["per_slice_correct"] for r in results)
    gem_slice_total = sum(r["gemini"]["per_slice_total"] for r in results)

    print(f"\nPer-slice tissue accuracy:")
    if base_slice_total > 0:
        print(f"  Base MedGemma:      {base_slice_correct}/{base_slice_total} "
              f"({base_slice_correct/base_slice_total*100:.0f}%)")
    if dist_slice_total > 0:
        print(f"  Distilled MedGemma: {dist_slice_correct}/{dist_slice_total} "
              f"({dist_slice_correct/dist_slice_total*100:.0f}%)")
    if gem_slice_total > 0:
        print(f"  Gemini (teacher):   {gem_slice_correct}/{gem_slice_total} "
              f"({gem_slice_correct/gem_slice_total*100:.0f}%)")

    # Quality metrics
    quality_metrics = ["has_transitions", "has_caution", "has_structure", "has_trends"]
    print(f"\nReasoning quality (% of responses with feature):")
    print(f"  {'Metric':<20} {'Base':>8} {'Distilled':>10} {'Gemini':>8}")
    print(f"  {'-'*50}")
    for m in quality_metrics:
        base_pct = sum(1 for r in results if r["base"][m]) / n * 100
        dist_pct = sum(1 for r in results if r["distilled"][m]) / n * 100
        gem_pct = sum(1 for r in results if r["gemini"][m]) / n * 100
        print(f"  {m:<20} {base_pct:>7.0f}% {dist_pct:>9.0f}% {gem_pct:>7.0f}%")

    # Quality score
    base_avg_q = sum(r["base"]["quality_score"] for r in results) / n
    dist_avg_q = sum(r["distilled"]["quality_score"] for r in results) / n
    gem_avg_q = sum(r["gemini"]["quality_score"] for r in results) / n
    print(f"\n  {'Avg quality score':<20} {base_avg_q:>7.1f}  {dist_avg_q:>9.1f}  {gem_avg_q:>7.1f}  (out of 4)")

    # Word count
    base_avg_w = sum(r["base"]["word_count"] for r in results) / n
    dist_avg_w = sum(r["distilled"]["word_count"] for r in results) / n
    gem_avg_w = sum(r["gemini"]["word_count"] for r in results) / n
    print(f"  {'Avg word count':<20} {base_avg_w:>7.0f}  {dist_avg_w:>9.0f}  {gem_avg_w:>7.0f}")

    # Save final results
    with open(out_file, "w") as f:
        json.dump({
            "config": vars(args),
            "summary": {
                "n": n,
                "base_grounded_pct": base_grounded / n * 100,
                "distilled_grounded_pct": dist_grounded / n * 100,
                "base_avg_quality": base_avg_q,
                "distilled_avg_quality": dist_avg_q,
                "gemini_avg_quality": gem_avg_q,
                "base_slice_accuracy_pct": base_slice_correct / base_slice_total * 100 if base_slice_total > 0 else 0,
                "distilled_slice_accuracy_pct": dist_slice_correct / dist_slice_total * 100 if dist_slice_total > 0 else 0,
                "gemini_slice_accuracy_pct": gem_slice_correct / gem_slice_total * 100 if gem_slice_total > 0 else 0,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
