# train_mobilenetv2_split.py
import json, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TRAIN_DIR = "DATASET_SPLIT/train"
VAL_DIR   = "DATASET_SPLIT/val"
IMG_SIZE  = (224, 224)
BATCH     = 32
EPOCHS    = 15

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")

labels = {v: k for k, v in train_gen.class_indices.items()}
with open("labels.json", "w") as f: json.dump(labels, f)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])

cb = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint("asl_mobilenetv2.h5", monitor="val_accuracy", save_best_only=True)
]
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cb)
print("✅ Saved asl_mobilenetv2.h5")
