from validation.crop_validation_pixel import crop_validation_pixel
from validation.crop_validation_embedding import crop_validation_embedding

def validate_crop_image(img, reference_embeddings):
    ok, msg = crop_validation_pixel(img)
    if not ok:
        return False, msg

    ok, msg = crop_validation_embedding(img, reference_embeddings)
    if not ok:
        return False, msg

    return True, "Crop validated"
