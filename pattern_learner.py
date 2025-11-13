"""
pattern_learner.py

Simple CNN to learn chart patterns from labeled screenshots, and
predict pattern tags on new screenshots.

Folder structure:

train_screens/
    Short_Trend_Start/
    Short_Liquidity_Test/
    Downtrend_Exhaustion/
    Long_Pullback_Healthy/
    Long_Fail_Stand_Aside/
    No_Trade/

new_screens/
    *.png / *.jpg to classify

Usage in Colab:
    - Mount / clone the repo so these folders are visible
    - Run this script as a notebook
"""

import os
import csv
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

# ======================= CONFIG ======================= #

DATA_DIR = "train_screens"   # labeled screenshots
NEW_DIR = "new_screens"      # unlabeled screenshots to classify
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "chart_pattern_cnn.h5")

IMG_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
VALIDATION_SPLIT = 0.2
SEED = 123


# ======================= TRAINING ======================= #

def build_generators():
    """Create train/validation generators from folder structure."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
        horizontal_flip=True,
        zoom_range=0.05,
        width_shift_range=0.02,
        height_shift_range=0.02,
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
    )

    return train_gen, val_gen


def build_model(num_classes: int) -> tf.keras.Model:
    """Simple CNN; good enough to learn your labelled A/B/C/D/E patterns."""
    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model():
    """Train the model on your labeled screenshots."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_gen, val_gen = build_generators()
    num_classes = train_gen.num_classes

    print(f"Found {train_gen.samples} training images across "
          f"{num_classes} classes.")
    print("Class indices:", train_gen.class_indices)

    model = build_model(num_classes)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
    )

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final validation accuracy: {final_val_acc:.3f}")

    return model, train_gen.class_indices


# ======================= PREDICTION ======================= #

def load_trained_model():
    """Load a previously saved model and its class indices."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run train_model() first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Rebuild a generator just to recover class_indices mapping
    temp_gen, _ = build_generators()
    class_indices = temp_gen.class_indices
    return model, class_indices


def predict_on_folder(
    folder: str,
    output_csv: str = "pattern_predictions.csv"
):
    """
    Run the trained model on all images in `folder` and
    save predictions to CSV: filename, label, confidence.
    """
    model, class_indices = load_trained_model()
    idx_to_class = {v: k for k, v in class_indices.items()}

    rows = [("filename", "predicted_label", "confidence")]

    image_files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No images found in folder: {folder}")
        return

    for fname in image_files:
        path = os.path.join(folder, fname)
        img = load_img(path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        arr = arr[None, ...]  # batch dimension

        probs = model.predict(arr, verbose=0)[0]
        best_idx = int(probs.argmax())
        best_label = idx_to_class[best_idx]
        confidence = float(probs[best_idx])

        rows.append((fname, best_label, confidence))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved predictions for {len(image_files)} images to {output_csv}")


# ======================= MAIN ======================= #

if __name__ == "__main__":
    # 1) Train the model (comment out once you've already trained)
    model, class_indices = train_model()

    # 2) After you have a trained model, you can classify new screenshots:
    #    Uncomment this when you have some images in NEW_DIR.
    # predict_on_folder(NEW_DIR)
