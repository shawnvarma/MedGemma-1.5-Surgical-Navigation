"""Model loading, inference helpers, and MedGemma reasoning tools.

The key deduplication here is ``run_inference()`` which replaces 6-9
copy-pasted generate-then-decode blocks throughout the original code.
"""

import re
import time

import torch
import numpy as np
from PIL import Image

from config import (
    MODEL_ID, LORA_PATH, REASONING_LORA_PATH,
    TISSUE_LABELS, clean_model_response,
)
from state import STATE
from tools import (
    tool_compute_3d_volumes, score_full_trajectory, tool_atlas_lookup,
)


# ===================================================================
# Model loading
# ===================================================================

def load_model():
    """Load MedGemma 1.5 4B with 4-bit quantization."""
    from transformers import (
        AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig,
    )

    print("Loading MedGemma 1.5 4B...")
    STATE.processor = AutoProcessor.from_pretrained(MODEL_ID)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    STATE.model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    print(f"  MedGemma ready ({torch.cuda.memory_allocated() / 1e9:.1f} GB)")


def load_lora():
    """Load fine-tuned LoRA adapters: tissue classification + reasoning."""
    from peft import PeftModel

    if STATE.model is None:
        print("  Skipping LoRA - base model not loaded")
        return

    if LORA_PATH is None or not LORA_PATH.exists():
        print(f"  Tissue LoRA path not set or not found")
        return

    print(f"Loading tissue LoRA: {LORA_PATH.name}")
    try:
        STATE.lora_model = PeftModel.from_pretrained(
            STATE.model, str(LORA_PATH),
            adapter_name="tissue", is_trainable=False,
        )
        STATE.lora_model.eval()
        print("  Tissue LoRA loaded (78% accuracy)")
    except Exception as e:
        print(f"  Warning: Could not load tissue LoRA: {e}")
        STATE.lora_model = None
        return

    if REASONING_LORA_PATH is not None and REASONING_LORA_PATH.exists():
        print(f"Loading reasoning LoRA: {REASONING_LORA_PATH.name}")
        try:
            STATE.lora_model.load_adapter(
                str(REASONING_LORA_PATH), adapter_name="reasoning")
            STATE.lora_model.set_adapter("tissue")
            print("  Reasoning LoRA loaded (3.2/4 quality)")
        except Exception as e:
            print(f"  Warning: Could not load reasoning LoRA: {e}")
    else:
        print("  Reasoning LoRA path not set or not found")


# ===================================================================
# Consolidated inference helper
# ===================================================================

def run_inference(messages, max_new_tokens=512, repetition_penalty=1.2,
                  model=None, clean=True):
    """Run a single MedGemma inference pass.

    This replaces 6-9 duplicated generate-then-decode blocks throughout
    the codebase.

    Args:
        messages: Chat-format message list for ``apply_chat_template``.
        max_new_tokens: Generation length limit.
        repetition_penalty: Repetition penalty (default 1.2).
        model: Model to use.  Falls back to ``STATE.model``.
        clean: If True, apply ``clean_model_response`` to the output.

    Returns:
        Decoded response string, optionally cleaned.
    """
    if model is None:
        model = STATE.model
    if model is None or STATE.processor is None:
        return ""

    inputs = STATE.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )

    response = STATE.processor.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    del output
    torch.cuda.empty_cache()

    return clean_model_response(response) if clean else response


# ===================================================================
# Slice helpers (used by trajectory analysis)
# ===================================================================

def get_clean_flair_image(z: int) -> Image.Image:
    """Get a clean FLAIR slice (no crosshair) as an RGB PIL Image."""
    slice_2d = STATE.volume[:, :, z].T
    slice_2d = slice_2d - slice_2d.min()
    if slice_2d.max() > 0:
        slice_2d = (slice_2d / slice_2d.max() * 255).astype(np.uint8)
    else:
        slice_2d = slice_2d.astype(np.uint8)
    return Image.fromarray(slice_2d, mode='L').convert('RGB')


def compute_slice_label_for_prompt(z: int) -> str:
    """Compute GT per-slice label in the training-format prompt style."""
    if STATE.segmentation is None:
        return f"SLICE {z}: UNKNOWN"

    seg_slice = STATE.segmentation[:, :, z]
    tissues = []
    for label, info in TISSUE_LABELS.items():
        if label == 0:
            continue
        count = int(np.sum(seg_slice == label))
        if count > 0:
            area_cm2 = count / 100.0
            tissues.append(f"{info['name']} ({area_cm2:.1f} cm\u00b2)")

    if not tissues:
        return f"SLICE {z}: HEALTHY \u2014 no tumor tissue detected"

    total = sum(
        int(np.sum(seg_slice == l)) for l in [1, 2, 4]) / 100.0
    return f"SLICE {z}: {', '.join(tissues)} \u2014 total {total:.1f} cm\u00b2"


def parse_narration_per_slice(narration_text: str,
                              slice_numbers: list) -> tuple:
    """Parse model narration into per-slice text and a trailing summary.

    Returns:
        (per_slice_dict, summary_text) where per_slice_dict maps
        slice_number -> description text.
    """
    per_slice = {}
    summary = ""

    pattern = r'\*{0,2}[Ss][Ll][Ii][Cc][Ee]\s+(\d+)\*{0,2}\s*[:\-\u2014]?\s*'
    markers = list(re.finditer(pattern, narration_text))

    if not markers:
        return per_slice, narration_text.strip()

    for idx, match in enumerate(markers):
        slice_num = int(match.group(1))
        start = match.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) \
            else len(narration_text)
        text = narration_text[start:end].strip()
        text = re.sub(r'^\*{1,2}\s*', '', text)

        if idx == len(markers) - 1:
            parts = re.split(r'\n\s*\n', text, maxsplit=1)
            if len(parts) == 2:
                text = parts[0].strip()
                summary = parts[1].strip()
            else:
                lines = text.split('\n')
                if len(lines) > 1:
                    text = lines[0].strip()
                    summary = '\n'.join(lines[1:]).strip()

        sentences = re.split(r'(?<=[.!])\s+', text)
        if len(sentences) > 2:
            text = ' '.join(sentences[:2])
        per_slice[slice_num] = text

    return per_slice, summary


# ===================================================================
# Tool 5: MedGemma Reasoning
# ===================================================================

def tool_medgemma_reason(image: Image.Image, context: str,
                         question: str) -> str:
    """MedGemma clinical reasoning given verified scaffold context."""
    prompt = (
        "You are a medical imaging assistant analyzing brain MRI scans.\n\n"
        f"CONTEXT (from segmentation and atlas):\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Provide a helpful, accurate response based on the image and "
        "context provided."
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    return run_inference(messages, max_new_tokens=300)


# ===================================================================
# Tool 6: Trajectory Analysis (two-phase: base vs distilled)
# ===================================================================

def tool_trajectory_analyze(trajectory_points: list) -> dict:
    """Analyze a surgical trajectory: Base MedGemma vs Distillation Fine-Tune.

    Phase 1 runs both models on the same multi-image prompt with GT-injected
    slice labels and verified measurements.  The scaffold then enforces safety
    rules and produces the final recommendation.

    Returns dict with 'text', 'points', 'recommendation', 'narration_response'.
    """
    if STATE.model is None or STATE.processor is None:
        return {"text": "Model not loaded", "points": [],
                "recommendation": ""}

    if len(trajectory_points) < 2:
        return {"text": "Need at least 2 points for trajectory analysis.",
                "points": [], "recommendation": ""}

    # ---------------------------------------------------------------
    # Step 1: Full trajectory scoring
    # ---------------------------------------------------------------
    full_score = score_full_trajectory(trajectory_points)
    segment_warnings = full_score.get("warnings", [])
    tissue_breakdown = full_score.get("tissue_breakdown", {})

    critical_blockers = [
        w for w in segment_warnings
        if any(t in w.lower()
               for t in ["brain-stem", "brainstem", "brain stem"])
    ]

    # ---------------------------------------------------------------
    # Step 2: Per-point scaffold analysis (GT-only, no model calls)
    # ---------------------------------------------------------------
    scaffold_analysis = []
    gt_point_summaries = []

    for i, pt in enumerate(trajectory_points):
        tissue = pt['tissue']
        region = pt.get('region', 'Unknown')
        region_display = (region if region and region not in
                          ["Unknown", "Background", ""]
                          else "deep white matter")

        if tissue == "BACKGROUND":
            decision = "DO NOT RESECT - healthy brain tissue"
            symbol = "X"
        elif tissue == "EDEMA":
            decision = "CAUTION - functional tissue may be present"
            symbol = "!"
        elif tissue == "NCR_NET":
            decision = "Necrotic core - safe to resect"
            symbol = "ok"
        elif tissue == "ENHANCING":
            decision = "Active tumor - primary resection target"
            symbol = "ok"
        else:
            decision = "Unknown tissue"
            symbol = "?"

        is_eloquent = pt.get('is_eloquent', False)
        eloquent_info = pt.get('eloquent_info', 'critical function')
        eloquent_note = (f"\n   ELOQUENT: {eloquent_info}"
                         if is_eloquent else "")

        scaffold_analysis.append({
            "point": i + 1,
            "slice": pt['z'],
            "x": pt['x'],
            "y": pt['y'],
            "tissue": tissue,
            "region": region_display,
            "scaffold_decision": decision,
            "tissue_symbol": symbol,
            "eloquent_note": eloquent_note,
            "is_eloquent": is_eloquent,
            "eloquent_info": eloquent_info if is_eloquent else None,
        })

        point_label = ("Entry" if i == 0
                       else ("Target"
                             if i == len(trajectory_points) - 1
                             else f"Mid-{i}"))
        gt_point_summaries.append(
            f"- {point_label} (Slice {pt['z']}): "
            f"{tissue} at {region_display}")

    # ---------------------------------------------------------------
    # Step 3: Build multi-image narration prompt
    # ---------------------------------------------------------------
    narration_content = []
    for pt in trajectory_points:
        clean_img = get_clean_flair_image(pt['z'])
        narration_content.append({"type": "image", "image": clean_img})
        narration_content.append({
            "type": "text",
            "text": compute_slice_label_for_prompt(pt['z']),
        })

    vol_data = tool_compute_3d_volumes(trajectory_points[0]['z'])
    slice_list = ", ".join(str(pt['z']) for pt in trajectory_points)
    n = len(trajectory_points)

    if vol_data.get("has_tumor"):
        tv = vol_data["tissue_volumes_cm3"]
        depth = vol_data["last_slice"] - vol_data["first_slice"]
        measurements = (
            f"VERIFIED MEASUREMENTS (from segmentation):\n"
            f"- Total tumor volume: {vol_data['total_volume_cm3']:.1f} cm\u00b3\n"
            f"- Tumor depth: {depth}mm "
            f"(slice {vol_data['last_slice']} to {vol_data['first_slice']})\n"
            f"- Enhancing: {tv.get('ENHANCING', 0):.1f} cm\u00b3\n"
            f"- Necrotic: {tv.get('NCR_NET', 0):.1f} cm\u00b3\n"
            f"- Edema: {tv.get('EDEMA', 0):.1f} cm\u00b3\n\n"
            f"You are shown exactly {n} slices: {slice_list}.\n"
            f"The surgeon is working top-down (superior \u2192 inferior) "
            f"through these slices.\n\n"
            f"For EACH of the {n} slices shown above, write 1-2 sentences "
            f"describing:\n"
            f"- HEALTHY slices: confirm clean tissue, note proximity to "
            f"tumor boundary\n"
            f"- EDEMA slices: CAUTION \u2014 peritumoral zone, functional "
            f"tissue at risk\n"
            f"- ENHANCING/NECROTIC slices: resection target, describe tumor "
            f"burden trend\n"
            f"- Tissue transitions: flag when crossing from "
            f"healthy\u2192edema or edema\u2192tumor\n\n"
            f"Reference specific measurements (cm\u00b2, cm\u00b3, mm) from "
            f"the GT data when describing each slice.\n"
            f"End with a 1-2 sentence trajectory summary citing total volume "
            f"and depth.\n"
            f"Do NOT describe any slices other than the {n} shown. "
            f"Only reference tissue types present in the GT data above."
        )
    else:
        measurements = (
            f"Describe what you observe at each of the {n} slices "
            f"along this trajectory."
        )

    narration_content.append({"type": "text", "text": measurements})
    messages_narration = [{"role": "user", "content": narration_content}]

    # ---------------------------------------------------------------
    # Step 4: Run base and distilled models
    # ---------------------------------------------------------------
    use_distillation = (
        STATE.lora_model is not None and
        "reasoning" in getattr(STATE.lora_model, 'peft_config', {}))

    base_model = STATE.lora_model if STATE.lora_model is not None \
        else STATE.model

    # Pass 1: Base MedGemma (adapters disabled)
    if use_distillation:
        STATE.lora_model.disable_adapter_layers()

    t0 = time.time()
    base_narration = run_inference(
        messages_narration, max_new_tokens=512, model=base_model)
    base_elapsed = time.time() - t0

    # Pass 2: Distillation Fine-Tune (reasoning adapter)
    if use_distillation:
        STATE.lora_model.set_adapter("reasoning")
        STATE.lora_model.enable_adapter_layers()

        t0 = time.time()
        distilled_narration = run_inference(
            messages_narration, max_new_tokens=512, model=STATE.lora_model)
        distilled_elapsed = time.time() - t0

        STATE.lora_model.set_adapter("tissue")
    else:
        distilled_narration = base_narration
        distilled_elapsed = base_elapsed

    # ---------------------------------------------------------------
    # Step 5: Format output
    # ---------------------------------------------------------------
    concerning = [p for p in scaffold_analysis
                  if p['tissue'] in ['BACKGROUND', 'EDEMA']
                  or p['is_eloquent']]

    result = ""
    if critical_blockers:
        result += "CRITICAL - TRAJECTORY NOT VIABLE:\n"
        for b in critical_blockers:
            result += f"   {re.sub(r'\\s*\\(\\d+mm\\)', '', b)}\n"
        result += "\n"

    slice_numbers = [pt['z'] for pt in trajectory_points]
    dist_per_slice, dist_summary = parse_narration_per_slice(
        distilled_narration, slice_numbers)
    base_per_slice, base_summary = parse_narration_per_slice(
        base_narration, slice_numbers)

    STATE.last_distilled_per_slice = dict(dist_per_slice)

    result += (f"**PER-SLICE ANALYSIS** (Base: {base_elapsed:.1f}s | "
               f"Distilled: {distilled_elapsed:.1f}s)\n")

    for pa in scaffold_analysis:
        z = pa['slice']
        label = ("Entry" if pa['point'] == 1
                 else ("Target" if pa['point'] == len(scaffold_analysis)
                       else f"Point {pa['point']}"))

        result += (f"\n---\n### {pa['tissue_symbol']} {label} "
                   f"\u2014 Slice {z} ({pa['tissue']} at {pa['region']})\n")

        gt_label = compute_slice_label_for_prompt(z)
        result += f"\n**GT:** {gt_label}\n"
        result += f"**Scaffold:** {pa['scaffold_decision']}"
        if pa['eloquent_note']:
            result += pa['eloquent_note']
        result += "\n"

        result += (f"\n**Base MedGemma 1.5:** "
                   f"{base_per_slice.get(z, '_No per-slice output parsed_')}\n")

        if use_distillation:
            result += (
                f"\n**Distillation Fine-Tune:** "
                f"{dist_per_slice.get(z, '_No per-slice output parsed_')}\n")

    if base_summary or (use_distillation and dist_summary):
        result += "\n---\n### Trajectory Summary\n"
        result += (f"\n**Base MedGemma 1.5:** "
                   f"{base_summary or '_No summary_'}\n")
        if use_distillation:
            result += (f"\n**Distillation Fine-Tune:** "
                       f"{dist_summary or '_No summary_'}\n")

    if concerning:
        result += f"\n**{len(concerning)} point(s) require caution**\n"

    if not critical_blockers:
        non_crit = [w for w in segment_warnings if w not in critical_blockers]
        if non_crit:
            result += "\n**Path warnings:**\n"
            for w in non_crit:
                result += f"   {w}\n"

    # Final recommendation (scaffold-enforced)
    result += "\n---\n**RECOMMENDATION:**\n"
    if critical_blockers:
        rec = "DO NOT PROCEED - trajectory crosses critical structures."
        result += f"**DO NOT PROCEED** - trajectory crosses critical structures.\n"
        result += "Consider alternative approach that avoids brain stem.\n"
    elif tissue_breakdown.get("BACKGROUND", 0) > 0:
        bg = tissue_breakdown["BACKGROUND"]
        rec = f"NOT RECOMMENDED - path crosses {bg}mm of healthy brain tissue."
        result += (f"**NOT RECOMMENDED** - trajectory crosses {bg}mm "
                   "of healthy brain tissue.\n")
        result += ("Resection along this path would cause "
                   "permanent deficits.\n")
    elif concerning:
        rec = (f"PROCEED WITH CAUTION - {len(concerning)} point(s) "
               "require consideration.")
        result += (f"**PROCEED WITH CAUTION** - {len(concerning)} point(s) "
                   "require careful consideration.\n")
        result += "Review caution points above before proceeding.\n"
    elif tissue_breakdown.get("EDEMA", 0) > 5:
        rec = "CAUTION - edema traversal detected."
        result += ("**CAUTION** - edema traversal detected. "
                   "Consider intraoperative mapping.\n")
    else:
        rec = "Trajectory appears viable for tumor resection."
        result += "Trajectory appears viable for tumor resection.\n"

    result += (f"\n_Base: {base_elapsed:.1f}s | "
               f"Distilled: {distilled_elapsed:.1f}s_")

    return {
        "text": result,
        "points": scaffold_analysis,
        "recommendation": rec,
        "narration_response": distilled_narration,
    }
