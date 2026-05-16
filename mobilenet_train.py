"""
================================================================================
Sign Language Recognition — MobileNetV2 Transfer Learning
================================================================================
Description:
    Trains a MobileNetV2 model on the grayscale sign language dataset using
    a two-phase transfer learning strategy. In Phase 1, the MobileNetV2 base
    is frozen and only the classification head is trained. In Phase 2, the
    top 30 layers of the base are unfrozen and fine-tuned with a reduced
    learning rate to improve generalisation.

    The tf.data pipeline is used for efficient GPU utilisation, with
    per-batch augmentation applied during training to improve robustness.

Dataset:
    37 classes (ASL A–Z, digits 0–9, custom 'Next' gesture)
    55,500 grayscale images at 50×50 pixels
    80/20 train/validation split

Reference:
    This work extends the Random Forest baseline described in:
    "Affordable Real-Time Hand Gesture Detection Using Random Forest"
    Published in IJISRT (peer-reviewed, Google Scholar indexed)
    https://www.ijisrt.com/affordable-realtime-hand-gesture-detection-using-random-forest
================================================================================
"""

import os
import time
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.models import Model


# ── Hyperparameters & Configuration ─────────────────────────────────────────
# All training parameters are centralised here for reproducibility.

GRAY_DIR   = "data/grayscale"     # preprocessed 50×50 grayscale image directory
MODEL_PATH = "model_mobilenet.keras"  # output path for the best saved model
IMG_SIZE   = 224                  # MobileNetV2 expects 224×224 input
BATCH_SIZE = 64                   # larger batch → more stable gradients, better GPU utilisation
SEED       = 42                   # fixed seed for reproducible train/val split
AUTOTUNE   = tf.data.AUTOTUNE     # let TensorFlow optimise parallelism automatically


# ── GPU Configuration ────────────────────────────────────────────────────────
# Enable memory growth to prevent TensorFlow from allocating all GPU memory
# at startup, allowing other processes to share the GPU during training.

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
print("GPUs available:", gpus)


# ── Augmentation Pipeline ────────────────────────────────────────────────────
# Applied per-batch on the training set only. Each transformation introduces
# controlled variation to improve the model's robustness to real-world
# conditions (lighting changes, minor rotations, contrast variation).
# horizontal_flip is intentionally excluded — sign language gestures are
# directionally meaningful and flipping would corrupt the label.

def augment(image, label):
    """Apply random photometric and geometric augmentations to a training image."""
    image = tf.image.random_brightness(image, 0.2)          # simulate lighting variation
    image = tf.image.random_contrast(image, 0.8, 1.2)       # simulate camera contrast changes
    image = tf.image.pad_to_bounding_box(image, 10, 10,     # pad before crop for shift effect
                                          IMG_SIZE + 20, IMG_SIZE + 20)
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])  # random crop = implicit shift
    image = tf.image.rot90(image,                            # minor 90° rotation
                           tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.int32))
    return image, label


# ── Image Loading Function ───────────────────────────────────────────────────
# Reads JPEG files from disk, decodes to RGB tensor, resizes to model input
# dimensions, and normalises pixel values to [0, 1].
# Note: grayscale images are decoded as 3-channel (channels=3) since
# MobileNetV2 expects RGB input — the three channels will be identical
# but the model architecture requires this format.

def load_image(path, label):
    """Load and preprocess a single image from a file path."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)          # decode as 3-channel RGB
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])     # resize to 224×224
    image = tf.cast(image, tf.float32) / 255.0               # normalise to [0, 1]
    return image, label


# ── Dataset Construction ─────────────────────────────────────────────────────
# Collect all image paths and integer class labels from the grayscale
# directory. Class names are sorted alphabetically to ensure a consistent
# and reproducible label-to-index mapping across runs.

class_names   = sorted(os.listdir(GRAY_DIR))
class_indices = {name: i for i, name in enumerate(class_names)}
num_classes   = len(class_names)

all_paths, all_labels = [], []
for cls in class_names:
    folder = os.path.join(GRAY_DIR, cls)
    for f in os.listdir(folder):
        if f.endswith(".jpg"):
            all_paths.append(os.path.join(folder, f))
            all_labels.append(class_indices[cls])

# Convert integer labels to one-hot encoded vectors required for
# categorical cross-entropy loss computation.
all_labels_oh = tf.keras.utils.to_categorical(all_labels, num_classes)

# Stratified 80/20 train/validation split.
# The dataset is shuffled with a fixed seed before splitting to ensure
# reproducibility while maintaining class balance across both splits.
total      = len(all_paths)
val_size   = int(0.2 * total)
train_size = total - val_size

dataset  = tf.data.Dataset.from_tensor_slices((all_paths, all_labels_oh))
dataset  = dataset.shuffle(total, seed=SEED)
train_ds = dataset.skip(val_size)
val_ds   = dataset.take(val_size)

# ── tf.data Pipeline ─────────────────────────────────────────────────────────
# prefetch(AUTOTUNE) overlaps GPU computation with CPU data loading,
# ensuring the GPU is never idle waiting for the next batch.
# The validation set is cached in memory after the first epoch to
# eliminate repeated disk I/O during evaluation.

train_ds = (train_ds
    .map(load_image, num_parallel_calls=AUTOTUNE)  # parallel image loading
    .map(augment,    num_parallel_calls=AUTOTUNE)  # parallel augmentation
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))                           # prefetch next batch to GPU

val_ds = (val_ds
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()                                       # cache validation set after first epoch
    .prefetch(AUTOTUNE))

print(f"Classes: {num_classes} | Train samples: {train_size} | Val samples: {val_size}")

# Persist class-to-index mapping for use during inference.
with open("class_indices.pkl", "wb") as f:
    pickle.dump(class_indices, f)


# ── Model Architecture ───────────────────────────────────────────────────────
# MobileNetV2 is chosen as the base model for its favourable trade-off between
# accuracy and computational efficiency, making it suitable for deployment on
# resource-constrained systems (a key requirement in the original paper's
# ESP32-based edge inference pipeline).
#
# The classification head replaces MobileNetV2's default ImageNet classifier:
#   GlobalAveragePooling2D  — reduces spatial feature maps to a 1D vector
#   BatchNormalization      — stabilises training on grayscale→RGB converted input
#   Dropout(0.5)            — primary regularisation layer
#   Dense(256, ReLU)        — learned feature combination layer
#   Dropout(0.4)            — secondary regularisation
#   Dense(37, Softmax)      — output probabilities over 37 sign classes

base_model = MobileNetV2(
    weights="imagenet",               # initialise with ImageNet pretrained weights
    include_top=False,                # exclude original ImageNet classification head
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False          # freeze base during Phase 1

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = BatchNormalization()(x)
x      = Dropout(0.5)(x)
x      = Dense(256, activation="relu")(x)
x      = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ── Callbacks ────────────────────────────────────────────────────────────────
# EarlyStopping   — halts training if validation loss does not improve for
#                   3 consecutive epochs; restores the best weights found.
# ReduceLROnPlateau — halves the learning rate if validation loss plateaus
#                   for 2 epochs, allowing finer convergence.
# ModelCheckpoint — saves only the best model (by validation loss) to disk.

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        patience=2, factor=0.5, verbose=1),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True, verbose=1)
]


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Classification Head Training (Base Frozen)
# ════════════════════════════════════════════════════════════════════════════
# In Phase 1, all MobileNetV2 base layers are frozen. Only the newly added
# classification head is trained. This allows the head to learn a meaningful
# mapping from MobileNetV2 features to sign language classes without
# corrupting the pretrained base weights with large gradient updates.
# Learning rate: 0.001 (standard Adam default, appropriate for head-only training)

print("\n" + "=" * 60)
print("PHASE 1 — Classification head training (base frozen)")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

start    = time.time()
history1 = model.fit(train_ds, validation_data=val_ds,
                     epochs=5, callbacks=callbacks)

print(f"\nPhase 1 completed in {(time.time() - start) / 60:.1f} minutes")
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Phase 1 Validation Accuracy: {acc * 100:.2f}%")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fine-Tuning (Top 30 Layers Unfrozen)
# ════════════════════════════════════════════════════════════════════════════
# In Phase 2, the top 30 layers of MobileNetV2 are unfrozen and trained
# alongside the classification head. Lower layers (which capture generic
# low-level features such as edges and textures) remain frozen to preserve
# the pretrained representations that transfer well from ImageNet.
#
# The learning rate is reduced to 0.0001 (10× lower than Phase 1) to
# prevent catastrophic forgetting of the pretrained feature representations
# through overly large weight updates during fine-tuning.

print("\n" + "=" * 60)
print("PHASE 2 — Fine-tuning (top 30 MobileNetV2 layers unfrozen)")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False           # keep lower layers frozen

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable_count} / {len(model.layers)}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 10× lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

start    = time.time()
history2 = model.fit(train_ds, validation_data=val_ds,
                     epochs=10, callbacks=callbacks)

print(f"\nPhase 2 completed in {(time.time() - start) / 60:.1f} minutes")
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Phase 2 Validation Accuracy: {acc * 100:.2f}%")


# ── Save Training History ────────────────────────────────────────────────────
# Combine Phase 1 and Phase 2 histories into a single dictionary for use
# in benchmark visualisation. The phase1_epochs key marks the boundary
# between the two training phases in the training curve plot.

full_history = {
    "train_acc"     : history1.history["accuracy"]      + history2.history["accuracy"],
    "val_acc"       : history1.history["val_accuracy"]  + history2.history["val_accuracy"],
    "train_loss"    : history1.history["loss"]          + history2.history["loss"],
    "val_loss"      : history1.history["val_loss"]      + history2.history["val_loss"],
    "phase1_epochs" : len(history1.history["loss"])     # index of Phase 1 → Phase 2 transition
}

with open("mobilenet_history.pkl", "wb") as f:
    pickle.dump(full_history, f)

print("\n" + "=" * 60)
print("Training complete.")
print(f"  Model saved   → {MODEL_PATH}")
print("  History saved → mobilenet_history.pkl")
print("=" * 60)