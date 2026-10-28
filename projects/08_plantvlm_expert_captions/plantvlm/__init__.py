"""
plantvlm — a small, layered library for fine-tuning a vision-language model on
manually validated image captions.

Design in one paragraph
-----------------------
Everything is organised around two swap points and one guarantee.

    Swap point 1 — WHERE CAPTIONS COME FROM.  plantvlm.data.sources
        A CaptionSource yields validated (class, image_id, caption) records.
        Today: an Excel workbook. Tomorrow: a CSV, a database, a second crop.
        Nothing downstream knows or cares.

    Swap point 2 — WHICH MODEL IS BEING TUNED.  plantvlm.models
        A VLMAdapter knows one model family: how to load it, how to turn an
        (image, instruction, caption) into tensors, where its assistant turn
        begins, and how to decode a generation. Today: Qwen2.5-VL. Adding
        another VLM is one new file and one registry entry.

    The guarantee — EVERY RESULT IS TRACEABLE.  plantvlm.training.run
        A run is a directory. It contains the frozen config, the exact JSONL it
        trained on, the hash of the source workbook and of every image, its
        logs, its checkpoints, its predictions, and its evaluation. Nothing is
        overwritten; nothing is recomputed to make a chart.

Layer order (a layer may only import from layers above it):

    config / io          — configuration, hashing, JSONL
    data                 — sources, pairing, verification, splitting
    models               — VLM adapters
    training             — dataset, trainer, run manager
    inference            — generation
    evaluation           — metrics, retrieval, reporting
"""

__version__ = "1.0.0"
