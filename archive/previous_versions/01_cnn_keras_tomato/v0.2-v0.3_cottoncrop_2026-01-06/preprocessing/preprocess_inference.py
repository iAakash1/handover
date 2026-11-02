from preprocessing.quality_checks import quality_checks
from preprocessing.preprocess_training import preprocess_for_training

def preprocess_for_inference(img):
    valid, msg = quality_checks(img)
    if not valid:
        raise ValueError(msg)

    return preprocess_for_training(img)
