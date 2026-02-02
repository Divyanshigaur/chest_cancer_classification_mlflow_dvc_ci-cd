import os
import yaml
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ----------------------------
# Load parameters
# ----------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

EPOCHS = params["training"]["epochs"]
BATCH_SIZE = params["training"]["batch_size"]
LR = params["training"]["learning_rate"]
IMG_SIZE = params["data"]["img_size"]

DATA_DIR = "data/raw"


# ----------------------------
# Data generators
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)


# ----------------------------
# Model definition
# ----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ----------------------------
# MLflow tracking
# ----------------------------
mlflow.set_experiment("Chest_Cancer_Detection")

with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("img_size", IMG_SIZE)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    val_loss, val_acc = model.evaluate(val_gen)

    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    mlflow.keras.log_model(model, "model")
