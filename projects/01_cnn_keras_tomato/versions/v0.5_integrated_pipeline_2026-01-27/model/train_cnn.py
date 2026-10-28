import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from preprocessing.preprocess_training import preprocess_for_training
from preprocessing.augment_training import augment_for_training

def load_dataset(dataset_dir, img_size=(32, 32)):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Training preprocessing
            img = preprocess_for_training(img, size=img_size)
            images.append(img)
            labels.append(class_to_idx[cls])

            # Training augmentation
            augmented_imgs = augment_for_training(img)
            for aug in augmented_imgs:
                images.append(aug)
                labels.append(class_to_idx[cls])

    X = np.array(images, dtype="float32")
    y = to_categorical(labels, num_classes=len(class_names))

    return X, y, class_names

def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model(dataset_dir):
    IMG_SIZE = (32, 32)
    EPOCHS = 50
    BATCH_SIZE = 32

    print("Loading dataset...")
    X, y, class_names = load_dataset(dataset_dir, IMG_SIZE)

    print("Dataset loaded")
    print("Images shape:", X.shape)
    print("Classes:", class_names)

    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_cnn(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=len(class_names)
    )

    model.summary()


    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath="model/best_model.keras",  
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("Training started...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True
    )

    model.save("model/final_model.keras")
    print("Final model saved as model/final_model.keras")

    print("Training completed")
    return model, history, class_names

if __name__ == "__main__":
    DATASET_DIR = "/Users/kharulkamaaksha/Desktop/PlantVillage_split/train"
    train_model(DATASET_DIR)
