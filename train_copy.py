import os
import yaml
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shutil


# ----------------------------
# Load parameters
# ----------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

EPOCHS = params["training"]["epochs"]
BATCH_SIZE = params["training"]["batch_size"]
LR = params["training"]["learning_rate"]
IMG_SIZE = params["data"]["img_size"]

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"


# ----------------------------
# Data Generators (Training Only)
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


# ----------------------------
# Model Definition
# ----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ----------------------------
# MLflow Tracking
# ----------------------------
mlflow.set_experiment("Chest_Cancer_Detection")

run_name = f"CNN3Layer_Epoch{EPOCHS}_Img{IMG_SIZE}"

with mlflow.start_run(run_name=run_name):

    # Log training parameters
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("img_size", IMG_SIZE)

    # Save copy of training code
    shutil.copy("src/train.py", "train_copy.py")
    mlflow.log_artifact("train_copy.py")

    # ----------------------------
    # Model Training
    # ----------------------------
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    # Log final validation metrics
    val_loss, val_acc = model.evaluate(val_gen)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)

    # ----------------------------
    # Save Model
    # ----------------------------
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    mlflow.keras.log_model(model, "model")
