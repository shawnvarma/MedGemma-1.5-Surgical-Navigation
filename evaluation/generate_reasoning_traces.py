"""
Generate Gemini reasoning traces for MedGemma distillation training.

Loops through BraTS2021 patients, sends FLAIR slices + GT measurements
to Gemini, validates grounding, and saves traces for LoRA fine-tuning.

Usage:
    conda activate medgemma
    export GEMINI_API_KEY="your_key_here"

    # Mini run (5 patients) to check pricing
    python evaluation/generate_reasoning_traces.py --brats_dir /path/to/brats2021 --num_patients 5

    # Full run
    python evaluation/generate_reasoning_traces.py --brats_dir /path/to/brats2021

Environment variables:
    GEMINI_API_KEY  -- required, Gemini API key
    BRATS_DIR       -- path to BraTS2021 dataset (default: ./brats2021)
    OUTPUT_DIR      -- where to write traces (default: ./reasoning_traces)
"""

import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

BRATS_DIR = Path(os.environ.get("BRATS_DIR", "brats2021"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "reasoning_traces"))
MODEL_ID = "gemini-3-flash-preview"
SEED = 42

# BraTS labels
LABEL_NECROTIC = 1
LABEL_EDEMA = 2
LABEL_ENHANCING = 4
VOXEL_VOL_MM3 = 1.0  # 1mm isotropic


# ============================================================
# Data loading (reads from brats2021 per-patient dirs)
# ============================================================

def find_patients(brats_dir: Path) -> list[str]:
    """Find all patient IDs that have both FLAIR and segmentation."""
    patients = []
    for d in sorted(brats_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("BraTS2021_"):
            continue
        flair = list(d.glob("*_flair.nii*"))
        seg = list(d.glob("*_seg.nii*"))
        if flair and seg:
            patients.append(d.name)
    return patients


def load_volume(brats_dir: Path, patient_id: str, modality: str) -> np.ndarray:
    """Load a NIfTI volume for a patient."""
    patient_dir = brats_dir / patient_id
    files = list(patient_dir.glob(f"*_{modality}.nii*"))
    if not files:
        raise FileNotFoundError(f"No {modality} file for {patient_id}")
    return nib.load(str(files[0])).get_fdata()


def slice_to_image(volume: np.ndarray, z: int) -> Image.Image:
    """Extract an axial slice as a PIL Image with radiological orientation."""
    s = volume[:, :, z]
    s = np.rot90(s, k=1)
    s = np.flipud(s)
    smin, smax = s.min(), s.max()
    if smax > smin:
        s = (s - smin) / (smax - smin) * 255
    else:
        s = np.zeros_like(s)
    return Image.fromarray(s.astype(np.uint8), mode='L').convert('RGB')


# ============================================================
# Metrics computed directly from seg mask
# ============================================================

def compute_patient_metrics(seg: np.ndarray) -> dict:
    """Compute tumor metrics from a segmentation volume."""
    tumor_mask = (seg == LABEL_NECROTIC) | (seg == LABEL_EDEMA) | (seg == LABEL_ENHANCING)

    if not np.any(tumor_mask):
        return {"has_tumor": False}

    # Find tumor slice range
    tumor_slices = np.any(tumor_mask, axis=(0, 1))
    slice_indices = np.where(tumor_slices)[0]
    min_slice = int(slice_indices.min())
    max_slice = int(slice_indices.max())

    # Volumes in cm3 (voxels * 1mm3 / 1000)
    enhancing_vol = float(np.sum(seg == LABEL_ENHANCING)) * VOXEL_VOL_MM3 / 1000
    necrotic_vol = float(np.sum(seg == LABEL_NECROTIC)) * VOXEL_VOL_MM3 / 1000
    edema_vol = float(np.sum(seg == LABEL_EDEMA)) * VOXEL_VOL_MM3 / 1000
    total_vol = enhancing_vol + necrotic_vol + edema_vol

    return {
        "has_tumor": True,
        "min_slice": min_slice,
        "max_slice": max_slice,
        "total_volume_cm3": total_vol,
        "enhancing_volume_cm3": enhancing_vol,
        "necrotic_volume_cm3": necrotic_vol,
        "edema_volume_cm3": edema_vol,
    }


def compute_slice_areas(seg: np.ndarray, z: int) -> dict:
    """Compute tissue areas for one slice in cm2."""
    s = seg[:, :, z]
    # Apply same orientation as the image
    s = np.rot90(s, k=1)
    s = np.flipud(s)

    enhancing = float(np.sum(s == LABEL_ENHANCING)) / 100.0  # mm2 to cm2
    necrotic = float(np.sum(s == LABEL_NECROTIC)) / 100.0
    edema = float(np.sum(s == LABEL_EDEMA)) / 100.0

    tissues = []
    if enhancing > 0:
        tissues.append({"name": "ENHANCING", "area_cm2": enhancing})
    if necrotic > 0:
        tissues.append({"name": "NECROTIC", "area_cm2": necrotic})
    if edema > 0:
        tissues.append({"name": "EDEMA", "area_cm2": edema})

    total = enhancing + necrotic + edema
    has_tumor = total > 0

    return {"has_tumor": has_tumor, "tissues": tissues, "total_area_cm2": total}


# ============================================================
# Slice selection + GT injection
# ============================================================

def pick_slice_indices(metrics: dict, n_tumor: int = 3) -> list[int]:
    """Pick slices covering healthy -> tumor -> healthy trajectory."""
    min_s = metrics["min_slice"]
    max_s = metrics["max_slice"]
    tumor_range = max_s - min_s

    if tumor_range < n_tumor:
        tumor_slices = list(range(min_s, max_s + 1))
    else:
        tumor_slices = []
        for i in range(n_tumor):
            z = min_s + int(i * tumor_range / (n_tumor - 1))
            tumor_slices.append(z)

    margin_above = min(max_s + 10, 154)
    margin_below = max(min_s - 10, 0)

    all_slices = [margin_above] + tumor_slices + [margin_below]
    all_slices.sort(reverse=True)

    seen = set()
    result = []
    for z in all_slices:
        if z not in seen:
            seen.add(z)
            result.append(z)
    return result


def format_gt_label(seg: np.ndarray, z: int) -> str:
    """Format GT injection label for one slice."""
    info = compute_slice_areas(seg, z)
    if not info["has_tumor"]:
        return f"SLICE {z}: HEALTHY -- no tumor tissue detected"
    parts = [f"{t['name']} ({t['area_cm2']:.1f} cm2)" for t in info["tissues"]]
    return f"SLICE {z}: {', '.join(parts)} -- total {info['total_area_cm2']:.1f} cm2"


def build_prompt(metrics: dict) -> str:
    """Build the instruction prompt appended after images + labels."""
    depth = metrics["max_slice"] - metrics["min_slice"]
    return f"""VERIFIED MEASUREMENTS (from segmentation):
- Total tumor volume: {metrics['total_volume_cm3']:.1f} cm3
- Tumor depth: {depth}mm (slice {metrics['max_slice']} to {metrics['min_slice']})
- Enhancing: {metrics['enhancing_volume_cm3']:.1f} cm3
- Necrotic: {metrics['necrotic_volume_cm3']:.1f} cm3
- Edema: {metrics['edema_volume_cm3']:.1f} cm3

The surgeon is working top-down (superior to inferior) through these slices.
The trajectory includes healthy margin slices above and below the tumor.

For each slice, write 1-2 sentences describing:
- HEALTHY slices: confirm clean tissue, note proximity to tumor boundary
- EDEMA slices: CAUTION -- peritumoral zone, functional tissue at risk
- ENHANCING/NECROTIC slices: resection target, describe tumor burden trend
- Tissue transitions: flag when crossing from healthy->edema or edema->tumor

End with a 1-2 sentence trajectory summary.
Keep total response under 150 words. Only reference tissue types present in the GT data above."""


# ============================================================
# Grounding validation
# ============================================================

def validate_grounding(response_text: str, gt_data: dict) -> dict:
    """Check response only references GT tissue types and numbers."""
    text_lower = response_text.lower()
    issues = []

    negation_prefix = r'(?:no|without|absence\s+of|absent|lack(?:ing|s)?|not?\s+any|zero)\s+'

    def is_affirmative_mention(tissue_keyword: str) -> bool:
        matches = list(re.finditer(tissue_keyword, text_lower))
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
        "ENHANCING": is_affirmative_mention(r'enhanc'),
        "NECROTIC": is_affirmative_mention(r'necro'),
        "EDEMA": is_affirmative_mention(r'edema'),
    }

    for tissue, mentioned in tissue_mentions.items():
        if mentioned and tissue not in gt_data["tissues_present"]:
            issues.append(f"HALLUCINATION: mentions {tissue} but not in GT")

    # Number validation (5% tolerance)
    numbers_in_response = re.findall(r'(\d+\.?\d*)\s*(?:cm[23]|mm)', response_text)
    gt_numbers = set()
    gt_numbers.add(f"{gt_data['volume_cm3']:.1f}")
    gt_numbers.add(f"{gt_data['enhancing_volume_cm3']:.1f}")
    gt_numbers.add(f"{gt_data['necrotic_volume_cm3']:.1f}")
    gt_numbers.add(f"{gt_data['edema_volume_cm3']:.1f}")
    for slice_areas in gt_data["slice_areas"].values():
        for area in slice_areas.values():
            gt_numbers.add(f"{area:.1f}")
    depth = gt_data["slice_range"][1] - gt_data["slice_range"][0]
    gt_numbers.add(str(depth))

    for num_str in numbers_in_response:
        try:
            num = float(num_str)
            matched = False
            for gt_str in gt_numbers:
                gt_num = float(gt_str)
                if gt_num > 0 and abs(num - gt_num) / gt_num < 0.05:
                    matched = True
                    break
                elif gt_num == 0 and num == 0:
                    matched = True
                    break
            if not matched:
                issues.append(f"UNGROUNDED NUMBER: {num_str} not in GT values")
        except ValueError:
            pass

    return {"grounded": len(issues) == 0, "issues": issues}


def build_gt_data(seg: np.ndarray, metrics: dict, slice_indices: list[int]) -> dict:
    """Build GT reference for validation."""
    tissues_present = set()
    slice_areas = {}

    for z in range(metrics["min_slice"], metrics["max_slice"] + 1):
        info = compute_slice_areas(seg, z)
        for t in info["tissues"]:
            tissues_present.add(t["name"])

    for z in slice_indices:
        info = compute_slice_areas(seg, z)
        areas = {t["name"]: t["area_cm2"] for t in info["tissues"]}
        areas["_total"] = info["total_area_cm2"]
        slice_areas[z] = areas

    return {
        "tissues_present": tissues_present,
        "volume_cm3": metrics["total_volume_cm3"],
        "enhancing_volume_cm3": metrics["enhancing_volume_cm3"],
        "necrotic_volume_cm3": metrics["necrotic_volume_cm3"],
        "edema_volume_cm3": metrics["edema_volume_cm3"],
        "slice_areas": slice_areas,
        "slice_range": (metrics["min_slice"], metrics["max_slice"]),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces via Gemini")
    parser.add_argument("--brats_dir", type=str, default=str(BRATS_DIR),
                        help="Path to BraTS2021 dataset directory")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for traces")
    parser.add_argument("--num_patients", type=int, default=None,
                        help="Limit to N patients (mini run). Default: all.")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split to generate. Default: all.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between API calls (rate limiting)")
    args = parser.parse_args()

    brats_dir = Path(args.brats_dir)
    output_dir = Path(args.output_dir)

    # API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        return

    try:
        from google import genai
    except ImportError:
        print("ERROR: pip install google-genai")
        return

    client = genai.Client(api_key=api_key)

    # Find patients and create splits
    random.seed(args.seed)
    all_patients = find_patients(brats_dir)
    random.shuffle(all_patients)

    n_total = len(all_patients)
    n_train = int(n_total * 0.65)
    n_val = int(n_total * 0.16)

    splits = {
        "train": all_patients[:n_train],
        "val": all_patients[n_train:n_train + n_val],
        "test": all_patients[n_train + n_val:],
    }

    print("=" * 60)
    print("Reasoning Trace Generation")
    print(f"Model: {MODEL_ID}")
    print(f"BraTS dir: {brats_dir}")
    print(f"Total patients: {n_total}")
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print("=" * 60)

    # Determine which splits to process
    if args.split == "all":
        splits_to_run = ["train", "val", "test"]
    else:
        splits_to_run = [args.split]

    # Create output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Save split assignments (so we can reproduce)
    split_file = output_dir / "splits.json"
    if not split_file.exists():
        with open(split_file, 'w') as f:
            json.dump({k: v for k, v in splits.items()}, f, indent=2)
        print(f"Saved split assignments to {split_file}")

    for split_name in splits_to_run:
        patients = splits[split_name]

        # Apply --num_patients limit
        if args.num_patients is not None:
            patients = patients[:args.num_patients]

        # Resume support: load existing results and skip already-done patients
        out_path = output_dir / f"{split_name}.json"
        existing_results = []
        done_ids = set()
        if out_path.exists():
            with open(out_path) as f:
                existing_results = json.load(f)
            done_ids = {r["patient_id"] for r in existing_results}
            print(f"\n  Resuming: {len(done_ids)} patients already done")

        remaining = [p for p in patients if p not in done_ids]

        print(f"\n{'-' * 60}")
        print(f"SPLIT: {split_name} ({len(remaining)} new + {len(done_ids)} existing = {len(patients)} total)")
        print(f"{'-' * 60}")

        if not remaining:
            print(f"  All patients already processed, skipping.")
            continue

        results = list(existing_results)
        grounded_count = sum(1 for r in existing_results if r.get("grounded"))
        failed_count = 0
        skipped_count = 0
        consecutive_errors = 0

        for patient_id in tqdm(remaining, desc=split_name):
            try:
                flair = load_volume(brats_dir, patient_id, "flair")
                seg = load_volume(brats_dir, patient_id, "seg")
            except Exception as e:
                tqdm.write(f"  SKIP {patient_id}: {e}")
                skipped_count += 1
                continue

            metrics = compute_patient_metrics(seg)
            if not metrics["has_tumor"]:
                skipped_count += 1
                continue

            slice_indices = pick_slice_indices(metrics)
            gt_data = build_gt_data(seg, metrics, slice_indices)

            # Save slice images
            patient_img_dir = images_dir / patient_id
            patient_img_dir.mkdir(exist_ok=True)
            image_paths = []
            for z in slice_indices:
                img = slice_to_image(flair, z)
                img_path = patient_img_dir / f"slice_{z:03d}.png"
                img.save(img_path)
                image_paths.append(str(img_path))

            # Build Gemini content
            content_parts = []
            for i, z in enumerate(slice_indices):
                content_parts.append(slice_to_image(flair, z))
                content_parts.append(format_gt_label(seg, z))
            content_parts.append(build_prompt(metrics))

            # Call Gemini
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=content_parts,
                )
                resp_text = response.text.strip()
                consecutive_errors = 0  # Reset on success
            except Exception as e:
                error_msg = str(e)
                tqdm.write(f"  ERROR {patient_id}: {error_msg[:100]}")
                failed_count += 1
                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        tqdm.write(f"\n  QUOTA EXHAUSTED -- stopping early. Resume later.")
                        break
                time.sleep(args.delay)
                continue

            # Validate
            validation = validate_grounding(resp_text, gt_data)
            if validation["grounded"]:
                grounded_count += 1
            else:
                tqdm.write(f"  GROUNDING FAIL {patient_id}: {validation['issues']}")

            # Build training sample
            user_parts = []
            for i, z in enumerate(slice_indices):
                user_parts.append({"type": "image", "image_index": i})
                user_parts.append({"type": "text", "text": format_gt_label(seg, z)})
            user_parts.append({"type": "text", "text": build_prompt(metrics)})

            sample = {
                "patient_id": patient_id,
                "images": image_paths,
                "slice_indices": slice_indices,
                "conversations": [
                    {"role": "system", "content": "You are a surgical navigation assistant for brain tumor resection."},
                    {"role": "user", "content": user_parts},
                    {"role": "assistant", "content": resp_text},
                ],
                "metrics": {
                    "total_volume_cm3": metrics["total_volume_cm3"],
                    "enhancing_volume_cm3": metrics["enhancing_volume_cm3"],
                    "necrotic_volume_cm3": metrics["necrotic_volume_cm3"],
                    "edema_volume_cm3": metrics["edema_volume_cm3"],
                    "min_slice": metrics["min_slice"],
                    "max_slice": metrics["max_slice"],
                },
                "grounded": validation["grounded"],
                "grounding_issues": validation["issues"],
                "word_count": len(resp_text.split()),
                "model": MODEL_ID,
            }
            results.append(sample)

            # Save incrementally every 10 traces
            if len(results) % 10 == 0:
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2)

            time.sleep(args.delay)

        # Save results
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        grounded_of_total = f"{grounded_count}/{len(results)}" if results else "0/0"
        avg_words = sum(r["word_count"] for r in results) / len(results) if results else 0

        print(f"\n  Results: {len(results)} traces generated")
        print(f"  Grounded: {grounded_of_total}")
        print(f"  Skipped: {skipped_count}, Failed: {failed_count}")
        print(f"  Avg words: {avg_words:.0f}")
        print(f"  Saved to: {out_path}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "model": MODEL_ID,
        "seed": args.seed,
        "brats_dir": str(brats_dir),
        "total_patients": n_total,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "grounding_tolerance": 0.05,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
