"""PlantDx demo application (Streamlit).

A thin, presentation-only layer over the existing, frozen PlantDx inference
code. It never trains, evaluates, or regenerates anything — it loads a trained
adapter and captions uploaded leaf images, reusing
``plantdx.training.inference`` and ``plantdx.evaluation`` as-is.
"""
