"""
SoyNet XAI Package
==================
Explainable AI modules including Grad-CAM and visualization.
"""

from xai.gradcam import GradCAM, make_gradcam_heatmap
from xai.xai_visualizer import (
    visualize_gradcam,
    generate_xai_outputs,
    overlay_heatmap,
    visualize_batch_gradcam
)

__all__ = [
    'GradCAM',
    'make_gradcam_heatmap',
    'visualize_gradcam',
    'generate_xai_outputs',
    'overlay_heatmap',
    'visualize_batch_gradcam'
]
