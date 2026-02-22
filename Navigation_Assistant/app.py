"""MedGemma 1.5 Surgical Navigation Assistant -- Gradio application.

Entry point: ``python app.py``

This module wires the Gradio UI components to the event handlers defined
in ``handlers.py`` and loads data / models at startup.
"""

import gradio as gr
import nibabel as nib

from config import SCAN_PATH, SEG_PATH, IS_SPACES, DISPLAY_SIZE
from state import STATE
from tools import load_atlas
from inference import load_model, load_lora, parse_narration_per_slice
from rendering import (
    get_slice_image, get_coronal_image, get_sagittal_image,
    create_3d_segmentation_view, PLOTLY_AVAILABLE,
)
from handlers import (
    on_slice_change, on_overlay_toggle, on_distances_toggle,
    on_trajectory_click, on_mode_change, on_action_button_click,
    on_analyze_trajectory, on_trajectory_clear,
    on_trajectory_point_select, on_jump_to_target,
)


# ===================================================================
# Gradio UI
# ===================================================================

def create_interface():
    """Build and return the Gradio Blocks interface."""

    fullscreen_css = """
    <style>
    #plot-3d-container:fullscreen {
        background-color: rgb(30, 30, 30);
        padding: 20px;
    }
    #plot-3d-container:fullscreen .plot-container {
        height: 100vh !important;
        width: 100vw !important;
    }
    </style>
    """

    with gr.Blocks(title="MedGemma Surgical Navigation",
                   head=fullscreen_css) as demo:

        gr.Markdown("""
        # MedGemma 1.5 4B Surgical Navigation Assistant

        **Tool-augmented VLM for brain tumor resection guidance**

        1. **Click** on the MRI to set your position
           (scroll down for additional anatomical views)
        2. **Select a mode**: Plan Trajectory, Compare Models, or
           Free Question
        3. **Click the action button** for MedGemma 1.5 analysis
        """)

        # State
        crosshair_state = gr.State({"pos": None})
        trajectory_state = gr.State({"points": [], "active": True})
        target_state = gr.State({"x": None, "y": None, "z": None})

        with gr.Row():
            # ---------- Left column: views ----------
            with gr.Column(scale=2):
                slice_slider = gr.Slider(
                    minimum=0, maximum=154, value=77, step=1,
                    label="Scroll through axial slices")

                with gr.Row():
                    show_overlay = gr.Checkbox(
                        label="Show Tissue Masks (Ground Truth)",
                        value=True)
                    show_distances = gr.Checkbox(
                        label="Show Distances", value=True)

                image_display = gr.Image(
                    label="Axial View (Click to query)",
                    type="pil", interactive=True)

                with gr.Row():
                    coronal_display = gr.Image(
                        label="Coronal View", type="pil",
                        interactive=False)
                    sagittal_display = gr.Image(
                        label="Sagittal View", type="pil",
                        interactive=False)

                with gr.Accordion("3D Tumor View", open=True):
                    plot_3d = gr.Plot(label="3D Segmentation",
                                     elem_id="plot-3d-container")
                    if PLOTLY_AVAILABLE:
                        with gr.Row():
                            refresh_3d_btn = gr.Button(
                                "Refresh 3D View", size="sm")
                            fullscreen_3d_btn = gr.Button(
                                "Fullscreen", size="sm",
                                elem_id="fullscreen-3d-btn")
                    else:
                        gr.Markdown(
                            "*Plotly not available - 3D view disabled*")

            # ---------- Middle column: results ----------
            with gr.Column(scale=1):
                result_display = gr.Markdown(
                    value=(
                        "**Trajectory Mode Active**\n\nClick **5 points** "
                        "on different slices to plan a surgical path "
                        "(superior \u2192 inferior: healthy margin \u2192 "
                        "tumor \u2192 healthy margin).\n\nThe distillation "
                        "fine-tune is optimized for 5-point sequential "
                        "corridors.\n\nClick 'Analyze Trajectory' "
                        "when done."),
                    label="Query Result")

                gr.Markdown("---")

                trajectory_points_display = gr.Markdown(
                    value="*No trajectory points*",
                    label="Trajectory Points")

                trajectory_point_selector = gr.Dropdown(
                    choices=[], value=None, label="Jump to Point",
                    interactive=True, visible=False)

                with gr.Row():
                    analyze_trajectory_btn = gr.Button(
                        "Analyze Trajectory", variant="primary",
                        interactive=True, visible=True)
                    clear_trajectory_btn = gr.Button(
                        "Clear", interactive=True, visible=True)

            # ---------- Right column: mode selection ----------
            with gr.Column(scale=1):
                query_mode = gr.Dropdown(
                    choices=["Plan Trajectory", "Compare Models",
                             "Free Question"],
                    value="Plan Trajectory", label="Query Mode",
                    interactive=True)

                question_input = gr.Textbox(
                    label="Ask MedGemma",
                    value="What's going on at the position I clicked?",
                    placeholder="Type your question here...",
                    lines=2, visible=False)

                with gr.Row():
                    action_button = gr.Button(
                        "Analyze Trajectory", variant="primary")
                    jump_to_target_btn = gr.Button(
                        "Jump to Target", interactive=False,
                        visible=True)

                medgemma_output = gr.Markdown(
                    value="*MedGemma responses will appear here*",
                    label="MedGemma Response")

        # ===========================================================
        # Event wiring
        # ===========================================================

        slice_slider.change(
            fn=on_slice_change,
            inputs=[slice_slider, show_overlay, show_distances,
                    crosshair_state, trajectory_state, target_state],
            outputs=[image_display, coronal_display, sagittal_display])

        show_overlay.change(
            fn=on_overlay_toggle,
            inputs=[show_overlay, show_distances, slice_slider,
                    crosshair_state, trajectory_state, target_state],
            outputs=[image_display, coronal_display, sagittal_display])

        show_distances.change(
            fn=on_distances_toggle,
            inputs=[show_distances, show_overlay, slice_slider,
                    crosshair_state, trajectory_state, target_state],
            outputs=[image_display, coronal_display, sagittal_display])

        image_display.select(
            fn=on_trajectory_click,
            inputs=[slice_slider, show_overlay, show_distances,
                    crosshair_state, trajectory_state],
            outputs=[image_display, coronal_display, sagittal_display,
                     result_display, crosshair_state,
                     trajectory_state, trajectory_points_display,
                     trajectory_point_selector])

        query_mode.change(
            fn=on_mode_change,
            inputs=[query_mode, trajectory_state],
            outputs=[action_button, question_input, jump_to_target_btn,
                     trajectory_state, result_display,
                     analyze_trajectory_btn, clear_trajectory_btn,
                     trajectory_point_selector,
                     medgemma_output, trajectory_points_display])

        action_button.click(
            fn=on_action_button_click,
            inputs=[query_mode, slice_slider, crosshair_state,
                    question_input, show_overlay, show_distances,
                    trajectory_state],
            outputs=[medgemma_output, image_display, coronal_display,
                     sagittal_display, target_state,
                     jump_to_target_btn, plot_3d])

        # Trajectory buttons
        def _on_analyze_btn(trajectory_state, z, crosshair_state):
            """Wrapper that merges analysis into trajectory state
            and regenerates the 3-D view."""
            traj_pts = (trajectory_state.get("points")
                        if trajectory_state else None)
            pos = (crosshair_state.get("pos")
                   if crosshair_state else None)
            updated = (trajectory_state.copy() if trajectory_state
                       else {"points": [], "active": False})

            for text, data in on_analyze_trajectory(trajectory_state):
                narration = data.get("narration_response", "")
                analysis_pts = data.get("points", [])
                if traj_pts and narration:
                    slices = [p['z'] for p in traj_pts]
                    per_slice, _ = parse_narration_per_slice(
                        narration, slices)
                    merged = []
                    for i, pt in enumerate(traj_pts):
                        np_ = pt.copy()
                        st = per_slice.get(pt['z'], "")
                        if st:
                            np_["medgemma_analysis"] = st
                        elif analysis_pts:
                            for ap in analysis_pts:
                                if ap.get("point") == i + 1:
                                    np_["medgemma_analysis"] = ap.get(
                                        "scaffold_decision", "")
                                    break
                        np_["medgemma_guidance"] = np_.get(
                            "medgemma_analysis", "")
                        merged.append(np_)
                    updated = {
                        "points": merged,
                        "active": trajectory_state.get("active", False)}
                if narration:
                    updated["narration_response"] = narration

                plot = (create_3d_segmentation_view(
                    current_z=z, crosshair_pos=pos,
                    trajectory_points=updated.get("points", traj_pts))
                    if PLOTLY_AVAILABLE else None)
                yield text, plot, updated

        analyze_trajectory_btn.click(
            fn=_on_analyze_btn,
            inputs=[trajectory_state, slice_slider, crosshair_state],
            outputs=[medgemma_output, plot_3d, trajectory_state])

        clear_trajectory_btn.click(
            fn=on_trajectory_clear,
            inputs=[trajectory_state],
            outputs=[trajectory_state, trajectory_points_display,
                     trajectory_point_selector])

        trajectory_point_selector.change(
            fn=on_trajectory_point_select,
            inputs=[trajectory_point_selector, trajectory_state,
                    show_overlay, show_distances, crosshair_state],
            outputs=[slice_slider, image_display, coronal_display,
                     sagittal_display, crosshair_state, plot_3d])

        jump_to_target_btn.click(
            fn=on_jump_to_target,
            inputs=[target_state, show_overlay, show_distances,
                    crosshair_state, trajectory_state],
            outputs=[slice_slider, image_display, coronal_display,
                     sagittal_display, crosshair_state])

        # 3-D view buttons
        if PLOTLY_AVAILABLE:
            def _refresh_3d(z, cs, ts, qm):
                pos = cs.get("pos") if cs else None
                pts = ts.get("points") if ts else None
                return create_3d_segmentation_view(
                    current_z=z, crosshair_pos=pos,
                    trajectory_points=pts)

            refresh_3d_btn.click(
                fn=_refresh_3d,
                inputs=[slice_slider, crosshair_state,
                        trajectory_state, query_mode],
                outputs=[plot_3d])

            fullscreen_3d_btn.click(
                fn=lambda: None, inputs=[], outputs=[],
                js="""
                () => {
                    const el = document.querySelector(
                        '#plot-3d-container');
                    if (!el) return;
                    if (!document.fullscreenElement)
                        el.requestFullscreen().catch(() => {});
                    else
                        document.exitFullscreen();
                }
                """)

        # Initial images
        def _init_images():
            return (get_slice_image(77, show_overlay=True),
                    get_coronal_image(120, z=77, show_overlay=True),
                    get_sagittal_image(120, z=77, show_overlay=True))

        def _init_3d():
            return (create_3d_segmentation_view(current_z=77)
                    if PLOTLY_AVAILABLE else None)

        demo.load(fn=_init_images,
                  outputs=[image_display, coronal_display,
                           sagittal_display])
        if PLOTLY_AVAILABLE:
            demo.load(fn=_init_3d, outputs=[plot_3d])

    return demo


# ===================================================================
# Main
# ===================================================================

def main():
    """Load data and models, then launch the Gradio interface."""
    print("=" * 60)
    print("MedGemma Surgical Navigation - Gradio Interface")
    print("=" * 60)

    print(f"\nRunning on: "
          f"{'HuggingFace Spaces' if IS_SPACES else 'Local'}")

    # Load FLAIR volume
    print(f"\nLoading FLAIR: {SCAN_PATH}")
    if SCAN_PATH.exists():
        STATE.volume = nib.load(SCAN_PATH).get_fdata()
        print(f"  Shape: {STATE.volume.shape}")
    else:
        print("  WARNING: File not found!")

    # Load GT segmentation
    if SEG_PATH.exists():
        print(f"Loading segmentation: {SEG_PATH}")
        STATE.segmentation = nib.load(SEG_PATH).get_fdata()
        print("  Mask loaded")
    else:
        print("  WARNING: Segmentation not found!")

    # Atlas + model
    load_atlas()
    load_model()
    load_lora()

    # Launch
    print("\nStarting Gradio interface...")
    demo = create_interface()
    demo.launch(share=not IS_SPACES)


if __name__ == "__main__":
    main()
