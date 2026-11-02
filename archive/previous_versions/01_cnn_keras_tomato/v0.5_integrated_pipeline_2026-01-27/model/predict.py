import numpy as np

def predict_disease(
    model,
    processed_img,
    class_names,
    confidence_threshold=0.6
):
    """
    Predict crop disease from a preprocessed image.

    Args:
        model: Trained Keras model (.keras format)
        processed_img: Output of preprocess_for_inference()
        class_names: List of disease class names
        confidence_threshold: Minimum confidence to accept prediction

    Returns:
        dict: Prediction result
    """

    # Ensure batch dimension
    img = np.expand_dims(processed_img, axis=0)

    # Forward pass
    probabilities = model.predict(img, verbose=0)[0]

    class_id = int(np.argmax(probabilities))
    confidence = float(np.clip(np.max(probabilities), 0.0, 1.0))

    # Confidence-based rejection
    if confidence < confidence_threshold:
        return {
            "status": "uncertain",
            "confidence": confidence
        }

    return {
        "status": "success",
        "class_id": class_id,
        "disease": class_names[class_id],
        "confidence": confidence
    }
