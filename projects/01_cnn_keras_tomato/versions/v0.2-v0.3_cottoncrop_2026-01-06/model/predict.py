import numpy as np

def predict_disease(model, processed_img, class_names, threshold=0.6):
    img = np.expand_dims(processed_img, axis=0)
    probs = model.predict(img, verbose=0)[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    if confidence < threshold:
        return {
            "status": "uncertain",
            "confidence": confidence
        }

    return {
        "status": "success",
        "disease": class_names[idx],
        "confidence": confidence
    }
