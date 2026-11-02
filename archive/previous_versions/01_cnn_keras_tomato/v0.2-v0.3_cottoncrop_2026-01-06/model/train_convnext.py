import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def train_model(dataset_dir):

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    print("Preparing data generators (ConvNeXt)...")

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_data = train_gen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_data = val_gen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = train_data.num_classes

    print("Building ConvNeXt-Tiny model...")

    base_model = ConvNeXtTiny(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # -----------------------------
    # Stage 1: Feature Extraction
    # -----------------------------
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.LayerNormalization(),
        layers.Dense(256, activation="gelu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            "model/best_model_convnext.keras",
            save_best_only=True
        )
    ]

    print("Stage 1: Training classifier head...")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=callbacks
    )

    # -----------------------------
    # Stage 2: Fine-tuning
    # -----------------------------
    print("Stage 2: Fine-tuning ConvNeXt...")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=callbacks
    )

    print("ConvNeXt training complete.")
    return model
