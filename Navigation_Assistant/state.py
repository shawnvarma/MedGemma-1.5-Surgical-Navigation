"""Global application state -- loaded once at startup."""


class AppState:
    """Mutable singleton holding loaded data, model, and cached results.

    Attributes are set during ``main()`` in ``app.py``.  All modules
    import the single ``STATE`` instance from here.
    """

    volume = None          # 3-D FLAIR volume  (240, 240, 155)
    segmentation = None    # 3-D GT label mask  (240, 240, 155)
    model = None           # Base MedGemma model
    processor = None       # Corresponding processor / tokenizer
    lora_model = None      # PeftModel with LoRA adapters

    atlas_data = {
        "cortical": None, "cortical_labels": None,
        "subcortical": None, "subcortical_labels": None,
        "loaded": False,
    }

    # Per-slice narration cached from the last trajectory analysis
    # (bypasses Gradio state serialization)
    last_distilled_per_slice = {}


STATE = AppState()
