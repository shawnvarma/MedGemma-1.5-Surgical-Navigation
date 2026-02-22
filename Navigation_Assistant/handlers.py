"""Gradio event handlers -- click, slice change, mode switching, analysis.

All model inference calls go through ``inference.run_inference()`` to
avoid duplicating the generate-then-decode pattern.
"""

import time

import gradio as gr

from config import (
    DISPLAY_SIZE, SCALE, MAX_TRAJECTORY_POINTS,
    RESECTION_GUIDANCE, ELOQUENT_REGIONS,
    get_tissue_display_name, clean_model_response,
)
from state import STATE
from tools import (
    tool_mask_lookup, tool_atlas_lookup,
    tool_compute_distances, tool_compute_tumor_distances,
    tool_distance_to_tumor, tool_find_region, tool_find_tissue,
    tool_compute_3d_volumes, score_trajectory_segment,
)
from inference import (
    run_inference, tool_medgemma_reason, tool_trajectory_analyze,
    compute_slice_label_for_prompt,
)
from rendering import (
    get_slice_image, get_coronal_image, get_sagittal_image,
    draw_arrow_to_target, draw_arrow,
    get_trajectory_marked_image,
    create_3d_segmentation_view, create_crosshair_image_for_model,
    PLOTLY_AVAILABLE,
)


# ===================================================================
# Position query (orchestrates tools)
# ===================================================================

def query_position(x: int, y: int, z: int) -> tuple:
    """Orchestrate scaffold tools for a single position query.

    Returns ``(formatted_text, color)``.
    """
    mask = tool_mask_lookup(x, y, z)
    tissue = mask["tissue"]
    label = mask.get("label", 0)

    atlas = tool_atlas_lookup(x, y, z)
    region = atlas.get("region", "Unknown")
    is_eloquent = atlas.get("is_eloquent", False)
    eloquent_info = atlas.get("eloquent_info")

    tumor_dist = tool_distance_to_tumor(x, y, z)
    tissue_dist = tool_compute_distances(x, y, z, label) if label > 0 else None

    guidance = RESECTION_GUIDANCE.get(tissue, {
        "class": "UNKNOWN", "guidance": "Unknown tissue", "color": "white"})

    tissue_display = get_tissue_display_name(tissue)
    lines = [
        f"**Position:** ({x}, {y}) Slice {z}",
        f"**Tissue:** {tissue_display}",
        f"**[{guidance['class']}]** {guidance['guidance']}",
    ]

    if region and region not in ("Background", "Unknown"):
        lines.append(f"**Region:** {region}")
        if is_eloquent:
            lines.append(f"**ELOQUENT:** {eloquent_info}")

    lines.append("")

    if label == 2:  # EDEMA
        lines.append("**Inside Edema - Margins to edema edge:**")
        if tissue_dist and tissue_dist.get("inside_tissue"):
            for key, lbl in [("superior_mm", "Anterior"),
                              ("inferior_mm", "Posterior"),
                              ("right_mm", "Patient-L"),
                              ("left_mm", "Patient-R")]:
                if key in tissue_dist:
                    lines.append(f"  \u2022 {lbl}: {tissue_dist[key]:.0f}mm")

        lines.append("")
        if tumor_dist.get("inside_tumor"):
            lines.append("**Tumor also present at this location**")
        elif tumor_dist.get("no_tumor"):
            lines.append("**No tumor detected in scan**")
        else:
            d = tumor_dist.get("distance_mm", 0)
            lines.append(
                f"**Distance to tumor: {d:.0f}mm** "
                f"({tumor_dist.get('direction', '')})")

    elif label in (1, 4):  # NCR_NET or ENHANCING
        lines.append("**Inside Tumor - Margins to tumor edge:**")
        if tumor_dist.get("inside_tumor"):
            for key, lbl in [("anterior_mm", "Anterior"),
                              ("posterior_mm", "Posterior"),
                              ("patient_left_mm", "Patient-L"),
                              ("patient_right_mm", "Patient-R"),
                              ("superior_mm", "Superior"),
                              ("inferior_mm", "Inferior")]:
                if key in tumor_dist:
                    lines.append(
                        f"  \u2022 {lbl}: {tumor_dist[key]:.0f}mm")

    else:  # BACKGROUND
        if tumor_dist.get("no_tumor"):
            lines.append("**No tumor detected in scan**")
        else:
            d = tumor_dist.get("distance_mm", 0)
            direction = tumor_dist.get("direction", "")
            lines.append(f"**Distance to tumor: {d:.0f}mm** ({direction})")
            if d < 5:
                lines.append("  *Very close to tumor margin*")
            elif d < 10:
                lines.append("  *Near tumor boundary*")
            elif d < 20:
                lines.append("  *Within typical surgical approach corridor*")
            else:
                lines.append("  *Outside immediate surgical field*")

    return "\n".join(lines), guidance["color"]


# ===================================================================
# Navigation handlers
# ===================================================================

def _ortho_views(x, y, z, show_overlay, traj_points=None):
    """Regenerate coronal + sagittal views at *x, y, z*."""
    cor = get_coronal_image(
        y, z=z, crosshair_x=x, show_overlay=show_overlay,
        trajectory_points=traj_points if traj_points else None)
    sag = get_sagittal_image(
        x, z=z, crosshair_y=y, show_overlay=show_overlay,
        trajectory_points=traj_points if traj_points else None)
    return cor, sag


def _draw_target_arrows(img, cor, sag, target_state):
    """Draw distance-to-tumor arrows on all three views if target exists."""
    if target_state is None:
        return img, cor, sag
    from_x = target_state.get("from_x")
    to_x = target_state.get("x")
    if from_x is None or to_x is None:
        return img, cor, sag

    from_y = target_state["from_y"]
    from_z = target_state["from_z"]
    to_y = target_state["y"]
    to_z = target_state["z"]

    img = draw_arrow_to_target(img, from_x, from_y, to_x, to_y, SCALE)

    z_depth = STATE.volume.shape[2] if STATE.volume is not None else 155
    z_max = z_depth - 1
    cor_h = int(z_depth * (DISPLAY_SIZE / 240))
    cor = draw_arrow(cor,
                      (int(from_x * SCALE), int((z_max - from_z) * (cor_h / z_depth))),
                      (int(to_x * SCALE), int((z_max - to_z) * (cor_h / z_depth))))
    sag_h = cor_h
    sag = draw_arrow(sag,
                      (int(from_y * SCALE), int((z_max - from_z) * (sag_h / z_depth))),
                      (int(to_y * SCALE), int((z_max - to_z) * (sag_h / z_depth))))
    return img, cor, sag


def on_slice_change(z, show_overlay, show_distances, crosshair_state,
                    trajectory_state, target_state=None):
    """Handle slice slider change."""
    pos = crosshair_state.get("pos") if crosshair_state else None
    distances = crosshair_state.get("distances") if crosshair_state else None
    traj_pts = (trajectory_state.get("points", [])
                if trajectory_state else [])

    if pos:
        x, y = pos
        mask = tool_mask_lookup(x, y, z)
        distances = tool_compute_distances(x, y, z, mask.get("label", 0))

    img = get_slice_image(z, crosshair_pos=pos, show_overlay=show_overlay,
                          show_distances=show_distances, distances=distances)

    if pos:
        cor, sag = _ortho_views(pos[0], pos[1], z, show_overlay, traj_pts)
    else:
        cor, sag = _ortho_views(120, 120, z, show_overlay, traj_pts)

    img, cor, sag = _draw_target_arrows(img, cor, sag, target_state)
    return img, cor, sag


def on_image_click(z, show_overlay, show_distances, crosshair_state, evt: gr.SelectData):
    """Handle click on image to place crosshair and query tissue."""
    x = max(0, min(int(evt.index[0] / SCALE), 239))
    y = max(0, min(int(evt.index[1] / SCALE), 239))

    result_text, _ = query_position(x, y, z)
    mask = tool_mask_lookup(x, y, z)
    distances = tool_compute_distances(x, y, z, mask.get("label", 0))
    new_state = {"pos": (x, y), "distances": distances}

    img = get_slice_image(z, crosshair_pos=(x, y), show_overlay=show_overlay,
                          show_distances=show_distances, distances=distances)
    cor, sag = _ortho_views(x, y, z, show_overlay)
    return img, cor, sag, result_text, new_state


def on_overlay_toggle(show_overlay, show_distances, z, crosshair_state,
                      trajectory_state, target_state=None):
    """Handle overlay toggle."""
    pos = crosshair_state.get("pos") if crosshair_state else None
    distances = crosshair_state.get("distances") if crosshair_state else None
    traj_pts = (trajectory_state.get("points", [])
                if trajectory_state else [])

    img = get_slice_image(z, crosshair_pos=pos, show_overlay=show_overlay,
                          show_distances=show_distances, distances=distances)
    if pos:
        cor, sag = _ortho_views(pos[0], pos[1], z, show_overlay, traj_pts)
    else:
        cor, sag = _ortho_views(120, 120, z, show_overlay, traj_pts)

    img, cor, sag = _draw_target_arrows(img, cor, sag, target_state)
    return img, cor, sag


def on_distances_toggle(show_distances, show_overlay, z, crosshair_state,
                        trajectory_state, target_state=None):
    """Handle distances toggle."""
    pos = crosshair_state.get("pos") if crosshair_state else None
    distances = crosshair_state.get("distances") if crosshair_state else None
    traj_pts = (trajectory_state.get("points", [])
                if trajectory_state else [])

    img = get_slice_image(z, crosshair_pos=pos, show_overlay=show_overlay,
                          show_distances=show_distances, distances=distances)
    if pos:
        cor, sag = _ortho_views(pos[0], pos[1], z, show_overlay, traj_pts)
    else:
        cor, sag = _ortho_views(120, 120, z, show_overlay, traj_pts)

    img, cor, sag = _draw_target_arrows(img, cor, sag, target_state)
    return img, cor, sag


# ===================================================================
# Free Question handler
# ===================================================================

def on_ask_medgemma(z, crosshair_state, question, show_overlay=False,
                    show_distances=True):
    """Handle MedGemma free-form question (generator for live updates).

    Also handles location/distance queries by drawing arrows to targets.
    """
    if not question.strip():
        yield ("Please enter a question.", gr.update(), gr.update(),
               gr.update(), gr.update(), gr.update())
        return

    pos = crosshair_state.get("pos") if crosshair_state else None
    distances = crosshair_state.get("distances") if crosshair_state else None

    q_lower = question.lower()
    is_dist_query = any(p in q_lower for p in [
        "how far", "where is", "find the", "distance to",
        "locate", "point to", "show me", "how close"])
    is_tumor_dist = is_dist_query and any(
        k in q_lower for k in ["tumor", "cancer", "mass", "lesion"])

    # --- Tumor distance shortcut ---
    if is_tumor_dist and pos:
        x, y = pos
        td = tool_distance_to_tumor(x, y, z)
        img = get_slice_image(z, crosshair_pos=pos,
                              show_overlay=show_overlay,
                              show_distances=show_distances,
                              distances=distances)
        cor, sag = _ortho_views(x, y, z, show_overlay)

        if td.get("no_tumor"):
            yield ("No tumor detected in scan.", img, cor, sag,
                   gr.update(), gr.update(interactive=False))
            return

        if td.get("inside_tumor"):
            resp = "**Inside the tumor**\n\n**Margins to tumor edge:**\n"
            for k, l in [("anterior_mm", "Anterior"),
                         ("posterior_mm", "Posterior"),
                         ("patient_left_mm", "Patient-Left"),
                         ("patient_right_mm", "Patient-Right"),
                         ("superior_mm", "Superior"),
                         ("inferior_mm", "Inferior")]:
                if k in td:
                    resp += f"  \u2022 {l}: {td[k]:.0f}mm\n"

            mk = tool_mask_lookup(x, y, z)
            resp += (f"\n**Tissue type:** "
                     f"{get_tissue_display_name(mk.get('tissue', 'TUMOR'))}\n")
            resp += "\nGetting surgical guidance..."
            yield (resp, img, cor, sag,
                   gr.update(), gr.update(interactive=False))

            min_m = min(td.get(k, 999) for k in [
                "anterior_mm", "posterior_mm",
                "patient_left_mm", "patient_right_mm"])
            prompt = (
                f"In 1-2 sentences, what is the surgical significance "
                f"of being {min_m:.0f}mm from the tumor margin during "
                f"brain tumor resection? The tissue is {mk['tissue']}.")
            msgs = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}]}]
            clin = run_inference(msgs, max_new_tokens=150)
            resp = resp.replace("Getting surgical guidance...",
                                f"**Surgical guidance:** {clin}")
            yield (resp, img, cor, sag,
                   gr.update(), gr.update(interactive=False))
            return

        # Outside tumor
        dist = td.get("distance_mm", 0)
        nx, ny, nz = (td.get("nearest_x", x), td.get("nearest_y", y),
                       td.get("nearest_z", z))
        img = draw_arrow_to_target(img, x, y, nx, ny, SCALE)
        img, cor, sag = _draw_target_arrows(
            img, cor, sag,
            {"x": nx, "y": ny, "z": nz,
             "from_x": x, "from_y": y, "from_z": z})

        resp = f"**Distance to tumor: {dist:.0f}mm**\n"
        resp += f"Direction: {td.get('direction', '')}\n"
        if abs(nz - z) > 2:
            resp += (f"Nearest tumor point on slice {nz} "
                     f"({abs(nz - z):.0f}mm "
                     f"{'superior' if nz > z else 'inferior'})\n")
        resp += "\nGetting surgical guidance..."
        new_target = {"x": nx, "y": ny, "z": nz}
        yield (resp, img, cor, sag, new_target,
               gr.update(interactive=True))

        prompt = (f"In 1-2 sentences, what does it mean surgically to "
                  f"be {dist:.0f}mm away from a brain tumor when "
                  f"planning an approach?")
        msgs = [{"role": "user",
                 "content": [{"type": "text", "text": prompt}]}]
        clin = run_inference(msgs, max_new_tokens=150)
        resp = resp.replace("Getting surgical guidance...",
                            f"**Surgical guidance:** {clin}")
        yield (resp, img, cor, sag, new_target,
               gr.update(interactive=True))
        return

    # --- Non-tumor distance query ---
    target_result = None
    if is_dist_query and pos:
        x, y = pos
        region_kw = [
            "motor", "sensory", "broca", "wernicke", "visual", "language",
            "memory", "frontal", "temporal", "parietal", "occipital",
            "hippocampus", "cortex", "precentral", "postcentral"]
        tissue_kw = ["edema", "swelling", "enhancing", "necrotic"]

        for kw in region_kw:
            if kw in q_lower:
                target_result = tool_find_region(kw, x, y, z)
                target_result["type"] = "region"
                break
        if target_result is None or not target_result.get("found"):
            for kw in tissue_kw:
                if kw in q_lower:
                    target_result = tool_find_tissue(kw, x, y, z)
                    target_result["type"] = "tissue"
                    break

    if target_result and target_result.get("found"):
        tx = target_result["target_x"]
        ty = target_result["target_y"]
        tz = target_result["target_z"]
        dmm = target_result.get("distance_mm", 0)
        dirn = target_result.get("direction", "")

        img = get_slice_image(z, crosshair_pos=pos,
                              show_overlay=show_overlay,
                              show_distances=show_distances,
                              distances=distances)
        cor, sag = _ortho_views(pos[0], pos[1], z, show_overlay)
        img = draw_arrow_to_target(img, pos[0], pos[1], tx, ty, SCALE)
        img, cor, sag = _draw_target_arrows(
            img, cor, sag,
            {"x": tx, "y": ty, "z": tz,
             "from_x": pos[0], "from_y": pos[1], "from_z": z})

        new_target = {"x": tx, "y": ty, "z": tz}

        if target_result["type"] == "region":
            name = target_result.get("region_name", "Unknown")
            resp = f"**{name}**: {dmm:.0f}mm {dirn}"
            if target_result.get("is_eloquent"):
                resp += "\n**Eloquent cortex**"
            if abs(tz - z) > 5:
                resp += (f"\nTarget on slice {tz} ({abs(tz - z):.0f}mm "
                         f"{'superior' if tz > z else 'inferior'})")
            yield (resp + "\n\nGetting functional context...",
                   img, cor, sag, new_target, gr.update(interactive=True))

            fprompt = (f"In one sentence, what is the function of the "
                       f"{name} and what deficit occurs if damaged?")
            msgs = [{"role": "user",
                     "content": [{"type": "text", "text": fprompt}]}]
            func = run_inference(msgs, max_new_tokens=80)
            resp += f"\n\n**Function:** {func}"
        else:
            name = target_result.get("tissue_name", "Unknown")
            resp = f"**{name}**: {dmm:.0f}mm {dirn}"
            if abs(tz - z) > 5:
                resp += f"\nCentroid on slice {tz}"

        yield (resp, img, cor, sag, new_target,
               gr.update(interactive=True))
        return

    if is_dist_query and not pos:
        img = get_slice_image(z, show_overlay=show_overlay)
        yield ("Click on the image first to set your current position.",
               img, gr.update(), gr.update(), gr.update(),
               gr.update(interactive=False))
        return

    if is_dist_query and target_result and not target_result.get("found"):
        reason = target_result.get("reason",
                                    "Region/tissue not found in scan")
        img = get_slice_image(z, crosshair_pos=pos,
                              show_overlay=show_overlay,
                              show_distances=show_distances,
                              distances=distances)
        yield (f"Could not locate: {reason}", img, gr.update(),
               gr.update(), gr.update(), gr.update(interactive=False))
        return

    # --- Standard MedGemma query ---
    img = get_slice_image(z, crosshair_pos=pos, show_overlay=show_overlay,
                          show_distances=show_distances, distances=distances)
    yield ("**MedGemma is thinking...**", img, gr.update(), gr.update(),
           gr.update(), gr.update(interactive=False))

    context_lines = []
    if STATE.segmentation is not None:
        vol = tool_compute_3d_volumes(z)
        if vol.get("has_tumor"):
            context_lines.append(
                "VERIFIED 3D MEASUREMENTS (from GT mask - 100% accurate):")
            context_lines.append(
                f"- Total tumor volume: {vol['total_volume_cm3']:.2f} cm\u00b3")
            context_lines.append(
                f"- Tumor extent: slices {vol['first_slice']} to "
                f"{vol['last_slice']} ({vol['total_slices']} slices, "
                f"{vol['tumor_height_mm']:.0f}mm height)")
            context_lines.append(
                f"- Distance to inferior tumor boundary: "
                f"{vol['distance_to_inferior_mm']:.0f}mm")
            context_lines.append(
                f"- Distance to superior tumor boundary: "
                f"{vol['distance_to_superior_mm']:.0f}mm")
            context_lines.append(
                f"- Volume above current slice: "
                f"{vol['volume_above_cm3']:.2f} cm\u00b3")
            context_lines.append(
                f"- Volume below current slice: "
                f"{vol['volume_below_cm3']:.2f} cm\u00b3")
            context_lines.append(
                f"- Current position: {vol['percent_through']:.0f}% "
                f"through tumor (inferior to superior)")
            tv = vol['tissue_volumes_cm3']
            if tv:
                ts = ", ".join(
                    f"{k}: {v:.2f}" for k, v in tv.items() if v > 0)
                context_lines.append(f"- Tissue volumes (cm\u00b3): {ts}")
            context_lines.append("")
            context_lines.append(
                "IMPORTANT: Use ONLY these provided measurements. "
                "Do NOT calculate your own numbers.")
            context_lines.append("")

    if pos:
        x, y = pos
        mk = tool_mask_lookup(x, y, z)
        at = tool_atlas_lookup(x, y, z)
        dr = tool_compute_distances(x, y, z, mk.get("label", 0))

        tissue = mk['tissue']
        region = at.get('region', 'Unknown')
        context_lines.append(f"Position: ({x}, {y}) on slice {z}")
        context_lines.append(f"Tissue type at this location: {tissue}")

        interp = {"NCR_NET": "NECROTIC TUMOR CORE (dead tumor, safe to resect)",
                   "ENHANCING": "ACTIVE ENHANCING TUMOR (primary resection target)",
                   "EDEMA": "PERITUMORAL EDEMA (functional tissue may be present)",
                   "BACKGROUND": "NORMAL BRAIN TISSUE (do not resect)"}
        if tissue in interp:
            context_lines.append(f"  -> This is {interp[tissue]}")

        context_lines.append(f"Anatomical location: {region}")
        if at.get("is_eloquent"):
            context_lines.append(
                f"WARNING: ELOQUENT CORTEX - {at.get('eloquent_info')}")
            context_lines.append(
                "  -> Extra caution required even if tumor is present")

        if dr.get("inside_tissue"):
            context_lines.append(
                f"Distance to edge of {tissue}: "
                f"Ant {dr.get('superior_mm', 0):.0f}mm, "
                f"Post {dr.get('inferior_mm', 0):.0f}mm, "
                f"Patient-L {dr.get('right_mm', 0):.0f}mm, "
                f"Patient-R {dr.get('left_mm', 0):.0f}mm")

        td2 = tool_compute_tumor_distances(x, y, z)
        if td2.get("inside_tumor"):
            context_lines.append(
                f"INSIDE TUMOR - Distance to tumor boundary: "
                f"Ant {td2.get('anterior_mm', 0):.0f}mm, "
                f"Post {td2.get('posterior_mm', 0):.0f}mm, "
                f"Patient-L {td2.get('right_mm', 0):.0f}mm, "
                f"Patient-R {td2.get('left_mm', 0):.0f}mm")
        elif not td2.get("no_tumor"):
            context_lines.append(
                f"Distance to nearest tumor: "
                f"{td2.get('distance_mm', 0):.0f}mm "
                f"({td2.get('direction', '')})")
    else:
        context_lines.append(f"Viewing slice {z}")
        context_lines.append(
            "No position selected - click on image to select a point")

    context = "\n".join(context_lines)
    query_img = get_slice_image(z, crosshair_pos=pos, show_overlay=False)

    t0 = time.time()
    response = tool_medgemma_reason(query_img, context, question)
    elapsed = time.time() - t0

    tag = " [Grounded]" if context_lines else ""
    yield (f"**MedGemma{tag}** ({elapsed:.1f}s):\n\n{response}",
           img, gr.update(), gr.update(), gr.update(),
           gr.update(interactive=False))


# ===================================================================
# Trajectory handlers
# ===================================================================

def on_trajectory_toggle(trajectory_mode, trajectory_state):
    """Handle trajectory mode toggle."""
    if trajectory_mode:
        return (
            {"points": [], "active": True},
            "**Trajectory Mode Active**\n\nClick up to 5 points on "
            "different slices to plan a surgical path.\n\nClick "
            "'Analyze Trajectory' when done.",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(choices=[], value=None, visible=False),
        )
    return (
        {"points": [], "active": False},
        "*Trajectory mode disabled*",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(choices=[], value=None, visible=False),
    )


def on_trajectory_clear(trajectory_state):
    """Clear trajectory points."""
    return (
        {"points": [], "active": trajectory_state.get("active", False)},
        "Trajectory cleared. Click to add points.",
        gr.update(choices=[], value=None, visible=False),
    )


def build_trajectory_dropdown_choices(points):
    """Build dropdown choices for trajectory point selector."""
    choices = []
    for i, pt in enumerate(points):
        region = pt.get('region', 'Unknown')
        short = region[:15] if region and region != "Unknown" else ""
        if short:
            choices.append(
                f"Point {i+1}: Slice {pt['z']} - {pt['tissue']} ({short})")
        else:
            choices.append(
                f"Point {i+1}: Slice {pt['z']} - {pt['tissue']}")
    return choices


def format_trajectory_points(points):
    """Format trajectory points for display with path metrics."""
    if not points:
        return "*No points selected*"

    lines = [f"**Trajectory: {len(points)} point(s)**\n"]
    total_cost, total_length = 0.0, 0
    all_warnings = []

    for i, pt in enumerate(points):
        tissue_display = get_tissue_display_name(pt['tissue'])
        guidance = RESECTION_GUIDANCE.get(pt['tissue'], {})
        rc = guidance.get('class', '?')
        region = pt.get('region', 'Unknown')

        line = f"{i+1}. Slice {pt['z']}: **{tissue_display}** [{rc}]"
        if region and region not in ("Unknown", "Background"):
            line += f"  \n   *{region}*"
        if pt.get('is_eloquent'):
            line += " !!"
        lines.append(line)

        seg_sc = pt.get('segment_score')
        if seg_sc:
            total_cost += seg_sc.get('cost', 0)
            total_length += seg_sc.get('length_mm', 0)
            for w in seg_sc.get('warnings', []):
                if w not in all_warnings:
                    all_warnings.append(w)

    if len(points) >= 2:
        lines.append("")
        lines.append(f"**Path: {total_length}mm, Cost: {total_cost:.0f}**")
    if all_warnings:
        lines.append("")
        lines.append("**Warnings:**")
        for w in all_warnings:
            lines.append(f"!! {w}")
    return "\n".join(lines)


def on_trajectory_click(z, show_overlay, show_distances, crosshair_state,
                        trajectory_state, evt: gr.SelectData):
    """Handle click in trajectory-aware mode."""
    x = max(0, min(int(evt.index[0] / SCALE), 239))
    y = max(0, min(int(evt.index[1] / SCALE), 239))

    points = (trajectory_state.get("points", [])
              if trajectory_state else [])
    is_active = (trajectory_state.get("active", False)
                 if trajectory_state else False)

    if not is_active:
        result_text, _ = query_position(x, y, z)
        mk = tool_mask_lookup(x, y, z)
        dists = tool_compute_distances(x, y, z, mk.get("label", 0))
        cs = {"pos": (x, y), "distances": dists}
        img = get_slice_image(z, crosshair_pos=(x, y),
                              show_overlay=show_overlay,
                              show_distances=show_distances,
                              distances=dists)
        cor, sag = _ortho_views(
            x, y, z, show_overlay, points if points else None)
        dd = build_trajectory_dropdown_choices(points) if points else []
        return (img, cor, sag, result_text, cs, trajectory_state,
                format_trajectory_points(points),
                gr.update(choices=dd, visible=len(dd) > 0))

    if len(points) >= MAX_TRAJECTORY_POINTS:
        mk = tool_mask_lookup(x, y, z)
        dists = tool_compute_distances(x, y, z, mk.get("label", 0))
        cs = {"pos": (x, y), "distances": dists}
        img = get_slice_image(z, crosshair_pos=(x, y),
                              show_overlay=show_overlay,
                              show_distances=show_distances,
                              distances=dists)
        cor, sag = _ortho_views(x, y, z, show_overlay, points)
        dd = build_trajectory_dropdown_choices(points)
        return (img, cor, sag,
                f"Maximum {MAX_TRAJECTORY_POINTS} points reached. "
                "Click 'Analyze Trajectory' or 'Clear'.",
                cs, trajectory_state,
                format_trajectory_points(points),
                gr.update(choices=dd, visible=True))

    # Add new point
    mk = tool_mask_lookup(x, y, z)
    at = tool_atlas_lookup(x, y, z)
    dists = tool_compute_distances(x, y, z, mk.get("label", 0))
    marked = get_trajectory_marked_image(z, x, y)

    new_pt = {
        'x': x, 'y': y, 'z': z,
        'tissue': mk['tissue'],
        'region': at.get('region', 'Unknown'),
        'is_eloquent': at.get('is_eloquent', False),
        'eloquent_info': at.get('eloquent_info'),
        'distances': dists,
        'image': marked,
        'segment_score': None,
    }

    seg_warnings = []
    if points:
        prev = points[-1]
        seg_sc = score_trajectory_segment(
            (prev['x'], prev['y'], prev['z']), (x, y, z))
        new_pt['segment_score'] = seg_sc
        seg_warnings = seg_sc.get('warnings', [])

    points = points + [new_pt]
    new_traj = {"points": points, "active": True}

    tissue_display = get_tissue_display_name(mk['tissue'])
    guidance = RESECTION_GUIDANCE.get(mk['tissue'], {})
    result = f"**Trajectory Point {len(points)}** added\n\n"
    result += f"**Position:** ({x}, {y}) Slice {z}\n"
    result += f"**Tissue:** {tissue_display}\n"
    result += (f"**[{guidance.get('class', 'UNKNOWN')}]** "
               f"{guidance.get('guidance', '')}\n")
    region = at.get('region', 'Unknown')
    if region and region not in ("Background", "Unknown"):
        result += f"**Region:** {region}\n"
        if at.get('is_eloquent'):
            result += f"**ELOQUENT:** {at.get('eloquent_info')}\n"
    if seg_warnings:
        result += "\n**--- Segment Warnings ---**\n"
        for w in seg_warnings:
            result += f"!! {w}\n"

    cs = {"pos": (x, y), "distances": dists}
    img = get_slice_image(z, crosshair_pos=(x, y),
                          show_overlay=show_overlay,
                          show_distances=show_distances,
                          distances=dists)
    cor, sag = _ortho_views(x, y, z, show_overlay, points)
    dd = build_trajectory_dropdown_choices(points)

    return (img, cor, sag, result, cs, new_traj,
            format_trajectory_points(points),
            gr.update(choices=dd, visible=True))


def on_trajectory_point_select(selected_point, trajectory_state,
                               show_overlay, show_distances,
                               crosshair_state):
    """Jump to a selected trajectory point's slice."""
    if not selected_point or not trajectory_state:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                crosshair_state, gr.update())

    points = trajectory_state.get("points", [])
    if not points:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                crosshair_state, gr.update())

    try:
        idx = int(selected_point.split(":")[0].replace("Point ", "")) - 1
        if idx < 0 or idx >= len(points):
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    crosshair_state, gr.update())

        pt = points[idx]
        x, y, z = pt['x'], pt['y'], pt['z']

        if not pt.get('medgemma_analysis') and STATE.last_distilled_per_slice:
            text = STATE.last_distilled_per_slice.get(z, "")
            if text:
                pt['medgemma_analysis'] = text

        mk = tool_mask_lookup(x, y, z)
        dists = tool_compute_distances(x, y, z, mk.get("label", 0))
        cs = {"pos": (x, y), "distances": dists}

        img = get_slice_image(z, crosshair_pos=(x, y),
                              show_overlay=show_overlay,
                              show_distances=show_distances,
                              distances=dists)
        cor, sag = _ortho_views(x, y, z, show_overlay, points)
        plot_3d = (create_3d_segmentation_view(
            current_z=z, crosshair_pos=(x, y),
            trajectory_points=points, selected_point_idx=idx)
            if PLOTLY_AVAILABLE else None)

        return z, img, cor, sag, cs, plot_3d
    except (ValueError, IndexError):
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                crosshair_state, gr.update())


def on_jump_to_target(target_state, show_overlay, show_distances,
                      crosshair_state, trajectory_state):
    """Navigate to the target location from a distance query."""
    if not target_state:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                crosshair_state)

    tx = target_state.get("x")
    ty = target_state.get("y")
    tz = target_state.get("z")
    if tx is None or ty is None or tz is None:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                crosshair_state)

    mk = tool_mask_lookup(tx, ty, tz)
    dists = tool_compute_distances(tx, ty, tz, mk.get("label", 0))
    cs = {"pos": (tx, ty), "distances": dists}
    traj_pts = (trajectory_state.get("points", [])
                if trajectory_state else [])

    img = get_slice_image(tz, crosshair_pos=(tx, ty),
                          show_overlay=show_overlay,
                          show_distances=show_distances,
                          distances=dists)
    cor, sag = _ortho_views(tx, ty, tz, show_overlay, traj_pts)
    img, cor, sag = _draw_target_arrows(img, cor, sag, target_state)

    return tz, img, cor, sag, cs


# ===================================================================
# Analyze trajectory
# ===================================================================

def on_analyze_trajectory(trajectory_state):
    """Analyze trajectory (generator for live updates)."""
    points = (trajectory_state.get("points", [])
              if trajectory_state else [])

    if len(points) < 2:
        yield ("Need at least 2 points to analyze trajectory.",
               {"points": [], "recommendation": "",
                "narration_response": ""})
        return

    yield (f"**Analyzing {len(points)}-point trajectory...**\n\n"
           "Running inference on both base MedGemma and distillation "
           "fine-tune (this may take ~4 minutes)...",
           {"points": [], "recommendation": "",
            "narration_response": ""})

    result = tool_trajectory_analyze(points)
    yield (result["text"], {
        "points": result["points"],
        "recommendation": result["recommendation"],
        "narration_response": result.get("narration_response", ""),
    })


# ===================================================================
# Mode switching
# ===================================================================

def on_mode_change(mode, trajectory_state):
    """Handle query mode dropdown change."""
    btn_labels = {
        "Plan Trajectory": "Analyze Trajectory",
        "Free Question": "Ask MedGemma",
        "Compare Models": "Compare Models",
    }
    btn_text = btn_labels.get(mode, "Submit")
    show_text = (mode == "Free Question")
    show_jump = False

    if mode == "Plan Trajectory":
        new_traj = {"points": [], "active": True}
        result = (
            "**Trajectory Mode Active**\n\n"
            "Click **5 points** on different slices to plan a surgical "
            "path (superior \u2192 inferior: healthy margin \u2192 tumor "
            "\u2192 healthy margin).\n\n"
            "The distillation fine-tune is optimized for 5-point "
            "sequential corridors.\n\nClick 'Analyze Trajectory' "
            "when done.")
        traj_btn = True
        clear_btn = True
        show_traj = True
    elif mode == "Compare Models":
        new_traj = {"points": [], "active": False}
        result = (
            "**Compare Models**\n\nClick anywhere on the MRI to ask "
            "MedGemma 1.5 what tissue type is at that location.\n\n"
            "Compares: **Base MedGemma** vs **LoRA Fine-Tune** vs "
            "**GT Scaffold**")
        traj_btn = False
        clear_btn = False
        show_traj = False
    else:
        new_traj = {"points": [], "active": False}
        result = (f"**{mode}** selected\n\nClick on the image to set "
                  "your position, then click the button.")
        traj_btn = False
        clear_btn = False
        show_traj = False

    return (
        gr.update(value=btn_text),
        gr.update(visible=show_text),
        gr.update(visible=show_jump, interactive=False),
        new_traj,
        result,
        gr.update(interactive=traj_btn, visible=show_traj),
        gr.update(interactive=clear_btn, visible=show_traj),
        gr.update(choices=[], value=None, visible=False),
        "*Click a location on the MRI, then ask MedGemma about the slice*",
        "*No trajectory points*",
    )


# ===================================================================
# Main action button (mode-dependent)
# ===================================================================

def on_action_button_click(mode, z, crosshair_state, question,
                           show_overlay, show_distances,
                           trajectory_state):
    """Dispatch to the correct handler based on current mode."""
    pos = crosshair_state.get("pos") if crosshair_state else None

    if mode == "Plan Trajectory":
        traj_pts = (trajectory_state.get("points")
                    if trajectory_state else None)
        for text, data in on_analyze_trajectory(trajectory_state):
            plot = (create_3d_segmentation_view(
                current_z=z, crosshair_pos=pos,
                trajectory_points=traj_pts)
                if PLOTLY_AVAILABLE else None)
            yield (text, gr.update(), gr.update(), gr.update(),
                   gr.update(), gr.update(), plot)

    elif mode == "Free Question":
        for r in on_ask_medgemma(z, crosshair_state, question,
                                 show_overlay, show_distances):
            yield (r[0], r[1], r[2], r[3], r[4], r[5], gr.update())

    elif mode == "Compare Models":
        for r in on_compare_models(z, crosshair_state,
                                   show_overlay, show_distances):
            yield (r[0], r[1], r[2], r[3], r[4], r[5], gr.update())

    else:
        yield ("Unknown mode", gr.update(), gr.update(), gr.update(),
               gr.update(), gr.update(), gr.update())


# ===================================================================
# Distance to Tumor handler
# ===================================================================

def on_distance_to_tumor(z, crosshair_state, show_overlay, show_distances):
    """Handle Distance to Tumor mode."""
    pos = crosshair_state.get("pos") if crosshair_state else None
    distances = crosshair_state.get("distances") if crosshair_state else None

    if not pos:
        yield ("**Click on the image first** to set your position.",
               gr.update(), gr.update(), gr.update(),
               gr.update(), gr.update(interactive=False))
        return

    x, y = pos
    td = tool_distance_to_tumor(x, y, z)
    img = get_slice_image(z, crosshair_pos=pos, show_overlay=show_overlay,
                          show_distances=show_distances, distances=distances)
    cor, sag = _ortho_views(x, y, z, show_overlay)

    if td.get("no_tumor"):
        yield ("No tumor detected in scan.", img, cor, sag,
               gr.update(), gr.update(interactive=False))
        return

    if td.get("inside_tumor"):
        resp = "**Inside the tumor**\n\n**Margins to tumor edge:**\n"
        for k, l in [("anterior_mm", "Anterior"),
                     ("posterior_mm", "Posterior"),
                     ("patient_left_mm", "Patient-Left"),
                     ("patient_right_mm", "Patient-Right"),
                     ("superior_mm", "Superior"),
                     ("inferior_mm", "Inferior")]:
            if k in td:
                resp += f"  \u2022 {l}: {td[k]:.0f}mm\n"

        mk = tool_mask_lookup(x, y, z)
        tissue = mk.get("tissue", "TUMOR")
        resp += f"\n**Tissue type:** {get_tissue_display_name(tissue)}\n"

        if tissue == "ENHANCING":
            resp += ("\n**[GT]** Active enhancing tumor \u2014 "
                     "primary resection target")
        elif tissue == "NCR_NET":
            resp += ("\n**[GT]** Necrotic/non-enhancing core \u2014 "
                     "safe to resect")
        else:
            resp += f"\n**[GT]** Inside tumor tissue ({tissue})"

        yield (resp, img, cor, sag,
               gr.update(), gr.update(interactive=False))
        return

    # Outside tumor
    dist = td.get("distance_mm", 0)
    nx, ny, nz = (td.get("nearest_x", x), td.get("nearest_y", y),
                   td.get("nearest_z", z))

    mk = tool_mask_lookup(x, y, z)
    current_tissue = mk.get("tissue", "UNKNOWN")
    at = tool_atlas_lookup(x, y, z)

    img = draw_arrow_to_target(img, x, y, nx, ny, SCALE)
    new_target = {
        "x": nx, "y": ny, "z": nz,
        "from_x": x, "from_y": y, "from_z": z,
    }
    img, cor, sag = _draw_target_arrows(img, cor, sag, new_target)

    resp = f"**Distance to tumor: {dist:.0f}mm**\n"
    resp += f"Direction: {td.get('direction', '')}\n"
    if abs(nz - z) > 2:
        resp += (f"Nearest tumor point on slice {nz} ({abs(nz - z):.0f}mm "
                 f"{'superior' if nz > z else 'inferior'})\n")
    resp += f"\n**Current position:** {current_tissue} at "
    resp += f"{at.get('region', 'Unknown')}\n"

    scaffold = {"BACKGROUND": "**DO NOT RESECT** \u2014 healthy brain tissue",
                "EDEMA": "**CAUTION** \u2014 edema (functional tissue with swelling)",
                "NCR_NET": "**TUMOR** \u2014 Safe to resect",
                "ENHANCING": "**TUMOR** \u2014 Safe to resect"}
    resp += f"{scaffold.get(current_tissue, '? Unknown tissue')}\n"

    if at.get("is_eloquent"):
        resp += (f"**ELOQUENT CORTEX:** "
                 f"{at.get('eloquent_info', 'critical function')}\n")

    # MedGemma context only for edema (scaffold is sufficient otherwise)
    if (current_tissue == "EDEMA" and STATE.model is not None
            and STATE.processor is not None):
        resp += "\nGetting clinical context..."
        yield (resp, img, cor, sag, new_target,
               gr.update(interactive=True))

        prompt = (
            f"The surgeon is at a position that is peritumoral edema, "
            f"located {dist:.0f}mm from the nearest tumor. In 1-2 "
            f"sentences, what is the surgical implication?")
        msgs = [{"role": "user",
                 "content": [{"type": "text", "text": prompt}]}]
        clin = run_inference(msgs, max_new_tokens=350)
        resp = resp.replace("Getting clinical context...",
                            f"**[MedGemma 1.5]:** {clin}")

    yield (resp, img, cor, sag, new_target,
           gr.update(interactive=True))


# ===================================================================
# Compare Models handler
# ===================================================================

def on_compare_models(z, crosshair_state, show_overlay, show_distances):
    """Compare tissue identification: Base vs LoRA vs GT Scaffold."""
    pos = crosshair_state.get("pos") if crosshair_state else None
    distances = crosshair_state.get("distances") if crosshair_state else None

    if not pos:
        yield ("**Click on the image first** to set your position.",
               gr.update(), gr.update(), gr.update(),
               gr.update(), gr.update(interactive=False))
        return

    x, y = pos
    img = get_slice_image(z, crosshair_pos=pos, show_overlay=show_overlay,
                          show_distances=show_distances, distances=distances)
    cor, sag = _ortho_views(x, y, z, show_overlay)

    mk = tool_mask_lookup(x, y, z)
    gt_tissue = mk.get("tissue", "UNKNOWN")
    at = tool_atlas_lookup(x, y, z)
    region = at.get("region", "Unknown")

    gt_binary = ("TUMOR" if gt_tissue in ("NCR_NET", "ENHANCING")
                 else ("EDEMA" if gt_tissue == "EDEMA" else "BACKGROUND"))

    yield ("Analyzing...", img, cor, sag,
           gr.update(), gr.update(interactive=False))

    marked = create_crosshair_image_for_model(z, x, y)
    prompt = ("What type of tissue is at the marked location in this MRI? "
              "Answer with exactly one word: TUMOR or EDEMA.")

    use_adapter = STATE.lora_model is not None

    # --- Base MedGemma ---
    base_pred, base_time = "ERROR", 0.0
    if STATE.model is not None and STATE.processor is not None:
        if use_adapter:
            STATE.lora_model.set_adapter("tissue")
            STATE.lora_model.disable_adapter_layers()

        model = STATE.lora_model if use_adapter else STATE.model
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": marked},
            {"type": "text", "text": prompt},
        ]}]

        t0 = time.time()
        raw = run_inference(msgs, max_new_tokens=10, model=model, clean=False)
        base_time = time.time() - t0

        upper = raw.upper()
        base_pred = ("TUMOR" if "TUMOR" in upper
                     else ("EDEMA" if "EDEMA" in upper else "UNCLEAR"))

        if use_adapter:
            STATE.lora_model.enable_adapter_layers()

    # --- LoRA Fine-Tune ---
    lora_pred, lora_time = "NOT LOADED", 0.0
    if STATE.lora_model is not None and STATE.processor is not None:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": marked},
            {"type": "text", "text": prompt},
        ]}]

        t0 = time.time()
        raw = run_inference(msgs, max_new_tokens=10, model=STATE.lora_model)
        lora_time = time.time() - t0

        upper = raw.upper()
        lora_pred = ("TUMOR" if "TUMOR" in upper
                     else ("EDEMA" if "EDEMA" in upper
                           else ("BACKGROUND" if "BACKGROUND" in upper
                                 else "UNCLEAR")))

    base_ok = "\u2713" if base_pred == gt_binary else "\u2717"
    lora_ok = "\u2713" if lora_pred == gt_binary else "\u2717"

    resp = "**TISSUE IDENTIFICATION COMPARISON**\n"
    resp += f"Position: ({x}, {y}) Slice {z} \u2014 {region}\n\n"
    resp += (f"---\n**[Base MedGemma 1.5]** ({base_time:.1f}s) "
             f"\u2192 {base_pred} {base_ok}\n\n")
    resp += (f"---\n**[Fine-tuned LoRA]** ({lora_time:.1f}s) "
             f"\u2192 {lora_pred} {lora_ok}\n\n")
    resp += (f"---\n**[GT Scaffold]** (instant) "
             f"\u2192 {gt_binary} \u2713\n")
    resp += (f"> *Ground truth: {get_tissue_display_name(gt_tissue)}*\n\n")
    resp += "---\n"
    if base_pred != gt_binary or lora_pred != gt_binary:
        resp += ("*This comparison demonstrates why ground truth "
                 "segmentation in conjunction with the atlas is "
                 "essential for reliable surgical guidance.*")

    yield (resp, img, cor, sag,
           gr.update(), gr.update(interactive=False))
