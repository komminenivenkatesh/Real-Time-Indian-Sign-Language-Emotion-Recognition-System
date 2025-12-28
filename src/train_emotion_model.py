import json
import os

# optional heavy dependency
try:
    import tensorflow as tf
    from tensorflow import keras
    # prefer accessing layers via the tf.keras namespace to avoid import errors
    layers = tf.keras.layers
except Exception:
    try:
        import keras
        layers = keras.layers
        tf = None
    except Exception:
        tf = None
        keras = None
        layers = None


def main():
    if keras is None:
        raise ImportError("TensorFlow/Keras is required to run train_emotion_model.py — install 'tensorflow' to proceed.")

    # -------- paths --------
    ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
    TRAIN_DIR = os.path.join(ROOT, "data", "train")
    TEST_DIR = os.path.join(ROOT, "data", "test")
    # prefer top-level `models/` folder that's present in the repo
    MODEL_DIR = os.path.join(ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}. Prepare image folders for training.")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}. Prepare test images or set TEST_DIR correctly.")

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
    if os.path.isdir(TEST_DIR):
        test_ds = keras.utils.image_dataset_from_directory(
            TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False, label_mode="categorical"
        )
    else:
        print(f"Warning: Test directory not found: {TEST_DIR}. Skipping final evaluation.")
        test_ds = None

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Save label map
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)

    # Prefetch for speed (only when tensorflow is available)
    if tf is not None:
        AUTOTUNE = getattr(tf.data, "AUTOTUNE", None)
        if AUTOTUNE is not None:
            train_ds = train_ds.prefetch(AUTOTUNE)
            val_ds = val_ds.prefetch(AUTOTUNE)
            if test_ds is not None:
                test_ds = test_ds.prefetch(AUTOTUNE)
    else:
        # No tf backend — skip prefetching
        pass

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

    model.summary()
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

    if test_ds is not None:
        print("Evaluating on test set…")
        print(model.evaluate(test_ds, verbose=2))
    else:
        print("No test dataset available — evaluation skipped.")


if __name__ == "__main__":
    main()
