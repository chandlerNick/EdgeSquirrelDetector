import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
IMAGE_SIZE = (224,224)  # Proper resoltuion for the picam v2
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = "../../data/images"
MODEL_PATH = "../../models/squirrel_classifier.keras"
TFLITE_PATH = "../../models/squirrel_classifier.tflite"

def load_datasets(data_dir=DATA_DIR, val_split=0.2):
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    # Prefetch to improve performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall")
        ]
    )

    return model


def fine_tune_model(model, train_ds, val_ds, epochs=10):
    early_stop = EarlyStopping(
        monitor="val_loss",  # or "val_recall", "val_accuracy", etc.
        patience=3,  # stop if no improvement after 3 epochs
        restore_best_weights=True,  # rollback to best epoch weights
        verbose=1
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop])
    return model, history


def export_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Saved model to {path}")

def convert_to_tflite(model, out_path=TFLITE_PATH, quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {out_path}")


def main():
    train_ds, val_ds = load_datasets()
    model = build_model()
    model, _ = fine_tune_model(model, train_ds, val_ds)
    export_model(model)
    convert_to_tflite(model, quantize=True)

if __name__=="__main__":
    main()
