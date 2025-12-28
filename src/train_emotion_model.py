import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------- paths --------
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
TRAIN_DIR = os.path.join(ROOT, "data", "train")
TEST_DIR = os.path.join(ROOT, "data", "test")
MODEL_DIR = os.path.join(ROOT, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 15

# -------- datasets --------
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    label_mode="categorical",
    seed=42,
    validation_split=0.1,
    subset="training",
)
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    label_mode="categorical",
    seed=42,
    validation_split=0.1,
    subset="validation",
)
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False, label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Save label map
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -------- data aug --------
data_augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="augment",
)

# -------- model (MobileNetV2) --------
base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False  # warmup with frozen base

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = data_augment(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

ckpt_path = os.path.join(MODEL_DIR, "emotion_mnv2_best.keras")
callbacks = [
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
]

print(model.summary())
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Fine-tune: unfreeze some layers
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
history2 = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

# Final save
final_path = os.path.join(MODEL_DIR, "emotion_mnv2_final.keras")
model.save(final_path)
print("Saved:", final_path, "and best:", ckpt_path)

print("Evaluating on test set…")
print(model.evaluate(test_ds, verbose=2))
