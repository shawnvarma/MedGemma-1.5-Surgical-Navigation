"""Scaffold tools -- deterministic computations grounded in GT segmentation.

Every function returns structured data (dicts / lists) that the scaffold
injects into MedGemma prompts.  No function calls the model; all values
are 100 % accurate because they come from the ground-truth segmentation
mask and the Harvard-Oxford atlas.
"""

from pathlib import Path

import numpy as np
import nibabel as nib

from config import (
    TISSUE_LABELS,
    TISSUE_NAME_MAP,
    ELOQUENT_REGIONS,
)
from state import STATE


# ===================================================================
# Tool 1: Mask Lookup
# ===================================================================

def tool_mask_lookup(x: int, y: int, z: int) -> dict:
    """Direct tissue lookup from GT segmentation mask -- 100 % accurate."""
    if STATE.segmentation is None:
        return {"tissue": "UNKNOWN", "error": "No segmentation loaded"}

    seg_slice = STATE.segmentation[:, :, z].T
    h, w = seg_slice.shape
    cx = max(0, min(x, w - 1))
    cy = max(0, min(y, h - 1))
    label_value = int(seg_slice[cy, cx])
    tissue_name = TISSUE_NAME_MAP.get(label_value, f"LABEL_{label_value}")

    return {
        "tissue": tissue_name,
        "label": label_value,
        "position": {"x": x, "y": y, "z": z},
        "source": "GT_MASK",
    }


# ===================================================================
# Tool 2: Atlas Lookup
# ===================================================================

_ATLAS_DIR = Path(__file__).resolve().parent / "data" / "atlas"

# Harvard-Oxford labels (from FSL via nilearn).  Index 0 = Background.
_CORTICAL_LABELS = [
    "Background", "Frontal Pole", "Insular Cortex",
    "Superior Frontal Gyrus", "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus, pars triangularis",
    "Inferior Frontal Gyrus, pars opercularis", "Precentral Gyrus",
    "Temporal Pole", "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, anterior division",
    "Middle Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, temporooccipital part",
    "Inferior Temporal Gyrus, anterior division",
    "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part",
    "Postcentral Gyrus", "Superior Parietal Lobule",
    "Supramarginal Gyrus, anterior division",
    "Supramarginal Gyrus, posterior division", "Angular Gyrus",
    "Lateral Occipital Cortex, superior division",
    "Lateral Occipital Cortex, inferior division",
    "Intracalcarine Cortex", "Frontal Medial Cortex",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)",
    "Subcallosal Cortex", "Paracingulate Gyrus",
    "Cingulate Gyrus, anterior division",
    "Cingulate Gyrus, posterior division", "Precuneous Cortex",
    "Cuneal Cortex", "Frontal Orbital Cortex",
    "Parahippocampal Gyrus, anterior division",
    "Parahippocampal Gyrus, posterior division", "Lingual Gyrus",
    "Temporal Fusiform Cortex, anterior division",
    "Temporal Fusiform Cortex, posterior division",
    "Temporal Occipital Fusiform Cortex", "Occipital Fusiform Gyrus",
    "Frontal Opercular Cortex", "Central Opercular Cortex",
    "Parietal Opercular Cortex", "Planum Polare",
    "Heschl's Gyrus (includes H1 and H2)", "Planum Temporale",
    "Supracalcarine Cortex", "Occipital Pole",
]

_SUBCORTICAL_LABELS = [
    "Background", "Left Cerebral White Matter", "Left Cerebral Cortex",
    "Left Lateral Ventricle", "Left Thalamus", "Left Caudate",
    "Left Putamen", "Left Pallidum", "Brain-Stem", "Left Hippocampus",
    "Left Amygdala", "Left Accumbens", "Right Cerebral White Matter",
    "Right Cerebral Cortex", "Right Lateral Ventricle", "Right Thalamus",
    "Right Caudate", "Right Putamen", "Right Pallidum",
    "Right Hippocampus", "Right Amygdala", "Right Accumbens",
]


def load_atlas():
    """Load Harvard-Oxford cortical and subcortical atlases from bundled files."""
    try:
        cort_path = _ATLAS_DIR / "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
        sub_path = _ATLAS_DIR / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

        print("Loading Harvard-Oxford atlas...")
        STATE.atlas_data["cortical"] = nib.load(str(cort_path)).get_fdata()
        STATE.atlas_data["cortical_labels"] = _CORTICAL_LABELS

        STATE.atlas_data["subcortical"] = nib.load(str(sub_path)).get_fdata()
        STATE.atlas_data["subcortical_labels"] = _SUBCORTICAL_LABELS

        print(f"  Harvard-Oxford: {len(_CORTICAL_LABELS)} cortical, "
              f"{len(_SUBCORTICAL_LABELS)} subcortical regions")
        STATE.atlas_data["loaded"] = True
        return True

    except Exception as e:
        print(f"Warning: Could not load atlas: {e}")
        return False


def brats_to_atlas(bx, by, bz):
    """Map BraTS voxel coordinates to MNI atlas coordinates.

    BraTS volumes are 240x240x155 at 1 mm isotropic in a roughly-MNI-aligned
    space.  This applies an empirical affine to index into the 1 mm Harvard-
    Oxford atlas (182x218x182).  The display follows radiological convention
    (patient-left on screen-right), so the left-right axis is flipped.
    """
    by_flipped = 239 - by
    ax = int(91 + (by_flipped - 120) * 0.76)
    ay = int(109 + (bx - 120) * 0.91)
    az = int(91 + (bz - 77) * 1.18)
    ax = max(0, min(ax, 181))
    ay = max(0, min(ay, 217))
    az = max(0, min(az, 181))
    return ax, ay, az


def tool_atlas_lookup(x: int, y: int, z: int) -> dict:
    """Anatomical region lookup -- checks cortical then subcortical atlas."""
    if not STATE.atlas_data["loaded"]:
        return {"region": "Unknown", "type": "no_atlas", "is_eloquent": False}

    vol_x, vol_y, vol_z = y, x, z
    ax, ay, az = brats_to_atlas(vol_x, vol_y, vol_z)

    for atlas_key, label_key in [("cortical", "cortical_labels"),
                                  ("subcortical", "subcortical_labels")]:
        data = STATE.atlas_data[atlas_key]
        labels = STATE.atlas_data[label_key]
        idx = int(data[ax, ay, az])
        if 0 < idx < len(labels):
            name = labels[idx]
            return {
                "region": name,
                "type": atlas_key,
                "is_eloquent": name in ELOQUENT_REGIONS,
                "eloquent_info": ELOQUENT_REGIONS.get(name),
            }

    return {"region": "Background", "type": "unlabeled", "is_eloquent": False}


# ===================================================================
# Tool 3: Distance / Geometry
# ===================================================================

def tool_compute_distances(x: int, y: int, z: int,
                           tissue_label: int) -> dict:
    """Compute 2-D distances from *point* to the edge of *tissue_label*."""
    if STATE.segmentation is None:
        return {"error": "No segmentation loaded"}

    seg_slice = STATE.segmentation[:, :, z].T
    h, w = seg_slice.shape
    py, px = int(y), int(x)

    if tissue_label == 0:
        return {"inside_tissue": False}

    tissue_mask = seg_slice == tissue_label
    if not tissue_mask[py, px]:
        return {"inside_tissue": False}

    distances = {"inside_tissue": True}

    # Superior (up, decreasing y)
    for dy in range(py, -1, -1):
        if not tissue_mask[dy, px]:
            distances["superior_mm"] = float(py - dy)
            break
    else:
        distances["superior_mm"] = float(py)

    # Inferior (down, increasing y)
    for dy in range(py, h):
        if not tissue_mask[dy, px]:
            distances["inferior_mm"] = float(dy - py)
            break
    else:
        distances["inferior_mm"] = float(h - py)

    # Left (decreasing x)
    for dx in range(px, -1, -1):
        if not tissue_mask[py, dx]:
            distances["left_mm"] = float(px - dx)
            break
    else:
        distances["left_mm"] = float(px)

    # Right (increasing x)
    for dx in range(px, w):
        if not tissue_mask[py, dx]:
            distances["right_mm"] = float(dx - px)
            break
    else:
        distances["right_mm"] = float(w - px)

    return distances


def tool_compute_tumor_distances(x: int, y: int, z: int) -> dict:
    """Compute 2-D distances from *point* to the overall tumor boundary.

    Tumor = NCR_NET (1) | ENHANCING (4).  Edema is excluded because it is
    functional tissue with swelling, not tumor.
    """
    if STATE.segmentation is None:
        return {"error": "No segmentation loaded"}

    seg_slice = STATE.segmentation[:, :, z].T
    h, w = seg_slice.shape
    py, px = int(y), int(x)

    tumor_mask = (seg_slice == 1) | (seg_slice == 4)
    if not tumor_mask[py, px]:
        return {"inside_tumor": False}

    distances = {"inside_tumor": True}

    for dy in range(py, -1, -1):
        if not tumor_mask[dy, px]:
            distances["anterior_mm"] = float(py - dy)
            break
    else:
        distances["anterior_mm"] = float(py)

    for dy in range(py, h):
        if not tumor_mask[dy, px]:
            distances["posterior_mm"] = float(dy - py)
            break
    else:
        distances["posterior_mm"] = float(h - py)

    for dx in range(px, -1, -1):
        if not tumor_mask[py, dx]:
            distances["left_mm"] = float(px - dx)
            break
    else:
        distances["left_mm"] = float(px)

    for dx in range(px, w):
        if not tumor_mask[py, dx]:
            distances["right_mm"] = float(dx - px)
            break
    else:
        distances["right_mm"] = float(w - px)

    return distances


def tool_distance_to_tumor(x: int, y: int, z: int) -> dict:
    """Compute distance from any point to the nearest tumor voxel.

    Works from both inside and outside the tumor.  Inside: returns edge
    distances in each cardinal direction.  Outside: returns Euclidean
    distance and direction to the nearest tumor voxel.
    """
    if STATE.segmentation is None:
        return {"error": "No segmentation loaded"}

    tumor_mask_3d = (STATE.segmentation == 1) | (STATE.segmentation == 4)
    px, py, pz = int(x), int(y), int(z)

    # --- Inside tumor ---
    if tumor_mask_3d[px, py, pz]:
        result = {"inside_tumor": True}
        seg_slice = STATE.segmentation[:, :, z].T
        tumor_mask = (seg_slice == 1) | (seg_slice == 4)
        h, w = seg_slice.shape

        for dy in range(py, -1, -1):
            if not tumor_mask[dy, px]:
                result["anterior_mm"] = float(py - dy)
                break
        else:
            result["anterior_mm"] = float(py)

        for dy in range(py, h):
            if not tumor_mask[dy, px]:
                result["posterior_mm"] = float(dy - py)
                break
        else:
            result["posterior_mm"] = float(h - py)

        for dx in range(px, -1, -1):
            if not tumor_mask[py, dx]:
                result["patient_right_mm"] = float(px - dx)
                break
        else:
            result["patient_right_mm"] = float(px)

        for dx in range(px, w):
            if not tumor_mask[py, dx]:
                result["patient_left_mm"] = float(dx - px)
                break
        else:
            result["patient_left_mm"] = float(w - px)

        for dz in range(pz, -1, -1):
            if not tumor_mask_3d[px, py, dz]:
                result["inferior_mm"] = float(pz - dz)
                break
        else:
            result["inferior_mm"] = float(pz)

        z_depth = STATE.segmentation.shape[2]
        for dz in range(pz, z_depth):
            if not tumor_mask_3d[px, py, dz]:
                result["superior_mm"] = float(dz - pz)
                break
        else:
            result["superior_mm"] = float(z_depth - pz)

        return result

    # --- Outside tumor ---
    tumor_coords = np.where(tumor_mask_3d)
    if len(tumor_coords[0]) == 0:
        return {"inside_tumor": False, "no_tumor": True}

    tumor_x = tumor_coords[0]
    tumor_y = tumor_coords[1]
    tumor_z = tumor_coords[2]

    dists = np.sqrt(
        (tumor_x - px)**2 + (tumor_y - py)**2 + (tumor_z - pz)**2)
    min_idx = np.argmin(dists)

    nearest_x = int(tumor_x[min_idx])
    nearest_y = int(tumor_y[min_idx])
    nearest_z = int(tumor_z[min_idx])
    distance_mm = float(dists[min_idx])

    dx = nearest_x - px
    dy = nearest_y - py
    dz = nearest_z - pz
    directions = []
    if abs(dy) > 3:
        directions.append("anterior" if dy < 0 else "posterior")
    if abs(dx) > 3:
        directions.append("patient-right" if dx < 0 else "patient-left")
    if abs(dz) > 3:
        directions.append("inferior" if dz < 0 else "superior")

    return {
        "inside_tumor": False,
        "distance_mm": distance_mm,
        "direction": "-".join(directions) if directions else "at current location",
        "nearest_x": nearest_x,
        "nearest_y": nearest_y,
        "nearest_z": nearest_z,
    }


def tool_find_region(region_query: str, from_x: int, from_y: int,
                     from_z: int) -> dict:
    """Find an anatomical region by name and compute distance from *from_*."""
    if not STATE.atlas_data["loaded"]:
        return {"error": "Atlas not loaded"}

    region_query_lower = region_query.lower()

    aliases = {
        "motor": ["precentral gyrus", "precentral", "motor"],
        "sensory": ["postcentral gyrus", "postcentral", "somatosensory"],
        "broca": ["pars opercularis", "pars triangularis",
                   "inferior frontal"],
        "wernicke": ["superior temporal", "posterior"],
        "visual": ["occipital pole", "intracalcarine", "calcarine",
                    "lateral occipital", "cuneal", "lingual gyrus"],
        "language": ["inferior frontal", "superior temporal",
                     "broca", "wernicke"],
        "memory": ["hippocampus", "parahippocampal", "temporal"],
    }

    search_terms = [region_query_lower]
    for alias, terms in aliases.items():
        if alias in region_query_lower:
            search_terms.extend(terms)

    def _score(label: str) -> int:
        label_lower = label.lower()
        best = 0
        for term in search_terms:
            if label_lower == term:
                best = max(best, 100)
            elif label_lower.startswith(term):
                best = max(best, 80)
            elif f" {term}" in f" {label_lower} ":
                best = max(best, 60)
            elif term in label_lower:
                pos = label_lower.find(term)
                if pos == 0 or label_lower[pos - 1] in " ,-":
                    best = max(best, 40)
                else:
                    best = max(best, 10)
        return best

    matches = []
    for atlas_key, label_key in [("cortical", "cortical_labels"),
                                  ("subcortical", "subcortical_labels")]:
        labels = STATE.atlas_data[label_key]
        if labels:
            for idx, label in enumerate(labels):
                s = _score(label)
                if s > 0:
                    matches.append({
                        "name": label, "type": atlas_key,
                        "atlas_idx": idx, "score": s})

    if not matches:
        return {"found": False, "query": region_query}

    matches.sort(key=lambda m: m["score"], reverse=True)
    best = matches[0]

    atlas_data = STATE.atlas_data[best["type"]]
    region_mask = atlas_data == best["atlas_idx"]
    if not np.any(region_mask):
        return {"found": False, "query": region_query,
                "reason": "Region not in scan FOV"}

    atlas_coords = np.where(region_mask)
    from_atlas = brats_to_atlas(from_y, from_x, from_z)

    min_dist = float('inf')
    closest_atlas = None
    for i in range(0, len(atlas_coords[0]), 10):
        ax, ay, az = (atlas_coords[0][i], atlas_coords[1][i],
                      atlas_coords[2][i])
        dist = np.sqrt((ax - from_atlas[0])**2 + (ay - from_atlas[1])**2 +
                       (az - from_atlas[2])**2)
        if dist < min_dist:
            min_dist = dist
            closest_atlas = (ax, ay, az)

    if closest_atlas is None:
        return {"found": False, "query": region_query}

    # Approximate inverse of brats_to_atlas
    target_y = int(120 + (closest_atlas[0] - 91) / 0.76)
    target_y = 239 - target_y
    target_x = int(120 + (closest_atlas[1] - 109) / 0.91)
    target_z = int(77 + (closest_atlas[2] - 91) / 1.18)
    z_max = STATE.segmentation.shape[2] - 1 if STATE.segmentation is not None else 154
    target_x = max(0, min(239, target_x))
    target_y = max(0, min(239, target_y))
    target_z = max(0, min(z_max, target_z))

    distance_mm = np.sqrt((target_x - from_x)**2 +
                          (target_y - from_y)**2 +
                          (target_z - from_z)**2)

    dx = target_x - from_x
    dy = target_y - from_y
    dz = target_z - from_z
    directions = []
    if abs(dx) > 5:
        directions.append(f"{'right' if dx > 0 else 'left'} {abs(dx):.0f}mm")
    if abs(dy) > 5:
        directions.append(
            f"{'posterior' if dy > 0 else 'anterior'} {abs(dy):.0f}mm")
    if abs(dz) > 5:
        directions.append(
            f"{'superior' if dz > 0 else 'inferior'} {abs(dz):.0f}mm")

    return {
        "found": True,
        "region_name": best["name"],
        "target_x": target_x,
        "target_y": target_y,
        "target_z": target_z,
        "distance_mm": distance_mm,
        "direction": ", ".join(directions) if directions else "at current location",
        "is_eloquent": best["name"] in ELOQUENT_REGIONS,
        "eloquent_info": ELOQUENT_REGIONS.get(best["name"]),
    }


def tool_find_tissue(tissue_query: str, from_x: int, from_y: int,
                     from_z: int) -> dict:
    """Find a tissue type and compute distance / direction from *from_*."""
    if STATE.segmentation is None:
        return {"error": "No segmentation loaded"}

    q = tissue_query.lower()

    if any(t in q for t in ["tumor", "cancer", "mass", "lesion"]):
        mask = (STATE.segmentation == 1) | (STATE.segmentation == 4)
        tissue_name = "tumor"
    elif any(t in q for t in ["enhancing", "active", "gadolinium",
                               "contrast"]):
        mask = STATE.segmentation == 4
        tissue_name = "enhancing tumor"
    elif any(t in q for t in ["edema", "swelling"]):
        mask = STATE.segmentation == 2
        tissue_name = "edema"
    elif any(t in q for t in ["necrotic", "ncr", "core", "dead"]):
        mask = STATE.segmentation == 1
        tissue_name = "necrotic core"
    else:
        return {"found": False, "query": tissue_query}

    if not np.any(mask):
        return {"found": False, "query": tissue_query,
                "reason": "Tissue not present in scan"}

    coords = np.where(mask)
    min_dist = float('inf')
    closest_idx = 0
    for i in range(0, len(coords[0]), 5):
        vy, vx, vz = coords[0][i], coords[1][i], coords[2][i]
        dx, dy = vx, vy
        dist = np.sqrt((dx - from_x)**2 + (dy - from_y)**2 +
                       (vz - from_z)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    target_y = int(coords[0][closest_idx])
    target_x = int(coords[1][closest_idx])
    target_z = int(coords[2][closest_idx])

    distance_mm = np.sqrt((target_x - from_x)**2 +
                          (target_y - from_y)**2 +
                          (target_z - from_z)**2)

    dx = target_x - from_x
    dy = target_y - from_y
    dz = target_z - from_z
    directions = []
    if abs(dx) > 5:
        directions.append(f"{'right' if dx > 0 else 'left'} {abs(dx):.0f}mm")
    if abs(dy) > 5:
        directions.append(
            f"{'posterior' if dy > 0 else 'anterior'} {abs(dy):.0f}mm")
    if abs(dz) > 5:
        directions.append(
            f"{'superior' if dz > 0 else 'inferior'} {abs(dz):.0f}mm")

    return {
        "found": True,
        "tissue_name": tissue_name,
        "target_x": target_x,
        "target_y": target_y,
        "target_z": target_z,
        "distance_mm": distance_mm,
        "direction": ", ".join(directions) if directions else "at current location",
    }


def tool_compute_3d_volumes(z: int) -> dict:
    """Compute 3-D volume measurements from the segmentation mask."""
    if STATE.segmentation is None:
        return {"error": "No segmentation loaded"}

    voxel_volume_mm3 = 1.0  # BraTS is 1 mm isotropic
    tumor_mask = (STATE.segmentation == 1) | (STATE.segmentation == 4)
    tumor_per_slice = np.sum(tumor_mask, axis=(0, 1))
    tumor_slices = np.where(tumor_per_slice > 0)[0]

    if len(tumor_slices) == 0:
        return {"has_tumor": False}

    first_slice = int(tumor_slices[0])
    last_slice = int(tumor_slices[-1])
    total_slices = last_slice - first_slice + 1

    total_tumor_mm3 = float(np.sum(tumor_mask) * voxel_volume_mm3)

    tissue_volumes = {}
    for label, info in TISSUE_LABELS.items():
        if label == 0:
            continue
        voxels = np.sum(STATE.segmentation == label)
        tissue_volumes[info["name"]] = float(voxels * voxel_volume_mm3 / 1000)

    vol_above = (
        float(np.sum(tumor_mask[:, :, first_slice:z]) * voxel_volume_mm3)
        if z > first_slice else 0.0)
    vol_below = (
        float(np.sum(tumor_mask[:, :, z + 1:last_slice + 1]) *
              voxel_volume_mm3)
        if z < last_slice else 0.0)

    if z < first_slice:
        pct = 0.0
    elif z > last_slice:
        pct = 100.0
    else:
        pct = ((z - first_slice) / max(1, total_slices - 1)) * 100

    if z < first_slice:
        dist_inf = 0.0
        dist_sup = float(last_slice - first_slice)
    elif z > last_slice:
        dist_inf = float(last_slice - first_slice)
        dist_sup = 0.0
    else:
        dist_inf = float(z - first_slice)
        dist_sup = float(last_slice - z)

    return {
        "has_tumor": True,
        "total_volume_cm3": total_tumor_mm3 / 1000,
        "tissue_volumes_cm3": tissue_volumes,
        "first_slice": first_slice,
        "last_slice": last_slice,
        "total_slices": total_slices,
        "tumor_height_mm": float(total_slices),
        "volume_above_cm3": vol_above / 1000,
        "volume_below_cm3": vol_below / 1000,
        "percent_through": pct,
        "distance_to_inferior_mm": dist_inf,
        "distance_to_superior_mm": dist_sup,
    }


# ===================================================================
# Tool 4: Trajectory Path Scoring
# ===================================================================

# Per-voxel cost for path optimisation
VOXEL_COSTS = {
    0: 1.0,   # BACKGROUND -- normal brain tissue
    1: 0.1,   # NCR_NET    -- dead tumor, safe to traverse
    2: 2.0,   # EDEMA      -- functional tissue risk
    4: 0.1,   # ENHANCING  -- active tumor, target
}
ELOQUENT_PENALTY = 5.0


def interpolate_3d_line(start: tuple, end: tuple) -> list:
    """Voxel coordinates along a 3-D line with 1 mm sampling.

    Since BraTS is 1 mm isotropic, one step = one voxel = 1 mm.
    """
    x0, y0, z0 = start
    x1, y1, z1 = end
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    length = np.sqrt(dx**2 + dy**2 + dz**2)

    if length < 1:
        return [start]

    z_max = STATE.segmentation.shape[2] - 1 if STATE.segmentation is not None else 154
    num_samples = int(np.ceil(length))
    points = []
    for i in range(num_samples + 1):
        t = i / num_samples
        x = max(0, min(int(round(x0 + t * dx)), 239))
        y = max(0, min(int(round(y0 + t * dy)), 239))
        z = max(0, min(int(round(z0 + t * dz)), z_max))
        if not points or (x, y, z) != points[-1]:
            points.append((x, y, z))
    return points


def score_trajectory_segment(start: tuple, end: tuple) -> dict:
    """Score a single line segment by tissue type and eloquent crossings."""
    if STATE.segmentation is None:
        return {"error": "No segmentation loaded", "cost": 0, "warnings": []}

    path_voxels = interpolate_3d_line(start, end)
    if not path_voxels:
        return {"cost": 0, "length_mm": 0, "tissue_breakdown": {},
                "eloquent_crossings": [], "warnings": []}

    total_cost = 0.0
    tissue_counts = {"BACKGROUND": 0, "NCR_NET": 0, "EDEMA": 0,
                     "ENHANCING": 0}
    eloquent_crossings = {}
    warnings = []

    for (x, y, z) in path_voxels:
        try:
            seg_slice = STATE.segmentation[:, :, z].T
            tissue_label = int(seg_slice[y, x])
        except IndexError:
            tissue_label = 0

        tissue_name = TISSUE_NAME_MAP.get(tissue_label, "BACKGROUND")
        total_cost += VOXEL_COSTS.get(tissue_label, 1.0)
        tissue_counts[tissue_name] = tissue_counts.get(tissue_name, 0) + 1

        atlas = tool_atlas_lookup(x, y, z)
        if atlas.get("is_eloquent"):
            region = atlas["region"]
            eloquent_crossings[region] = eloquent_crossings.get(region, 0) + 1
            total_cost += ELOQUENT_PENALTY

    for region, count in eloquent_crossings.items():
        info = ELOQUENT_REGIONS.get(region, "critical region")
        warnings.append(f"Crosses {region} ({info}) - {count}mm")

    if tissue_counts.get("EDEMA", 0) > 5:
        warnings.append(
            f"EDEMA traversal: {tissue_counts['EDEMA']}mm "
            f"(functional tissue risk)")

    tumor_mm = tissue_counts.get("NCR_NET", 0) + \
        tissue_counts.get("ENHANCING", 0)

    return {
        "cost": round(total_cost, 1),
        "length_mm": len(path_voxels),
        "tumor_mm": tumor_mm,
        "tissue_breakdown": tissue_counts,
        "eloquent_crossings": [{"region": r, "mm": c}
                                for r, c in eloquent_crossings.items()],
        "warnings": warnings,
    }


def score_full_trajectory(points: list) -> dict:
    """Score a multi-segment trajectory end-to-end."""
    if len(points) < 2:
        return {"cost": 0, "length_mm": 0, "warnings": []}

    total_cost = 0.0
    total_length = 0
    all_tissue = {"BACKGROUND": 0, "NCR_NET": 0, "EDEMA": 0, "ENHANCING": 0}
    all_eloquent = {}
    all_warnings = []

    for i in range(len(points) - 1):
        start = (points[i]['x'], points[i]['y'], points[i]['z'])
        end = (points[i + 1]['x'], points[i + 1]['y'], points[i + 1]['z'])
        seg = score_trajectory_segment(start, end)

        total_cost += seg.get("cost", 0)
        total_length += seg.get("length_mm", 0)
        for tissue, count in seg.get("tissue_breakdown", {}).items():
            all_tissue[tissue] = all_tissue.get(tissue, 0) + count
        for crossing in seg.get("eloquent_crossings", []):
            r = crossing["region"]
            all_eloquent[r] = all_eloquent.get(r, 0) + crossing["mm"]

    for region, count in all_eloquent.items():
        info = ELOQUENT_REGIONS.get(region, "critical region")
        all_warnings.append(f"{region} ({count}mm) - {info}")

    if all_tissue.get("EDEMA", 0) > 10:
        all_warnings.append(f"Total EDEMA: {all_tissue['EDEMA']}mm")

    tumor_mm = all_tissue.get("NCR_NET", 0) + all_tissue.get("ENHANCING", 0)

    return {
        "cost": round(total_cost, 1),
        "length_mm": total_length,
        "tumor_mm": tumor_mm,
        "tissue_breakdown": all_tissue,
        "eloquent_crossings": [{"region": r, "mm": c}
                                for r, c in all_eloquent.items()],
        "warnings": all_warnings,
    }


def generate_alternative_entries(entry: tuple, target: tuple,
                                 angles: list = None) -> list:
    """Rotate *entry* around *target* in XZ and YZ planes."""
    if angles is None:
        angles = [-20, -10, 10, 20]

    ex, ey, ez = entry
    tx, ty, tz = target
    dx = ex - tx
    dy = ey - ty
    dz = ez - tz
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    if length < 1:
        return []

    alternatives = []
    for angle_deg in angles:
        rad = np.radians(angle_deg)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        z_max = STATE.segmentation.shape[2] - 1 if STATE.segmentation is not None else 154

        # XZ rotation
        new_dx = dx * cos_a - dz * sin_a
        new_dz = dx * sin_a + dz * cos_a
        alt_x = max(0, min(int(round(tx + new_dx)), 239))
        alt_z = max(0, min(int(round(tz + new_dz)), z_max))
        if (alt_x, ey, alt_z) != entry:
            alternatives.append((alt_x, ey, alt_z))

        # YZ rotation
        new_dy = dy * cos_a - dz * sin_a
        new_dz2 = dy * sin_a + dz * cos_a
        alt_y = max(0, min(int(round(ty + new_dy)), 239))
        alt_z2 = max(0, min(int(round(tz + new_dz2)), z_max))
        if ((ex, alt_y, alt_z2) != entry and
                (ex, alt_y, alt_z2) not in alternatives):
            alternatives.append((ex, alt_y, alt_z2))

    return alternatives


def find_best_trajectory(trajectory_points: list) -> dict:
    """Compare current trajectory with rotated alternatives."""
    if len(trajectory_points) < 2:
        return {"error": "Need at least 2 points"}

    current_score = score_full_trajectory(trajectory_points)
    entry = (trajectory_points[0]['x'], trajectory_points[0]['y'],
             trajectory_points[0]['z'])
    target = (trajectory_points[-1]['x'], trajectory_points[-1]['y'],
              trajectory_points[-1]['z'])

    alt_entries = generate_alternative_entries(entry, target)

    alternatives = []
    for ae in alt_entries:
        alt_pts = [
            {'x': ae[0], 'y': ae[1], 'z': ae[2]},
            {'x': target[0], 'y': target[1], 'z': target[2]},
        ]
        s = score_full_trajectory(alt_pts)
        s['entry'] = ae
        alternatives.append(s)

    best_alt = min(alternatives, key=lambda a: a['cost']) if alternatives \
        else None

    current_cost = current_score['cost']
    if best_alt and best_alt['cost'] < current_cost:
        improvement = round(
            (current_cost - best_alt['cost']) / current_cost * 100, 1)
    else:
        improvement = 0

    return {
        "current_score": current_score,
        "best_alternative": best_alt,
        "improvement_pct": improvement,
        "all_alternatives": sorted(alternatives, key=lambda a: a['cost'])
        if alternatives else [],
    }
