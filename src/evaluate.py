import os
import yaml
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, confusion_matrix


# ----------------------------
# Load parameters
# ----------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

BATCH_SIZE = params["training"]["batch_size"]
IMG_SIZE = params["data"]["img_size"]

VAL_DIR = "data/val"
MODEL_PATH = "models/model.h5"


# ----------------------------
# Load trained model
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH)


# ----------------------------
# Validation Data Generator
# ----------------------------
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


# ----------------------------
# MLflow Tracking
# ----------------------------
mlflow.set_experiment("Chest_Cancer_Detection")

with mlflow.start_run(run_name="Evaluation_Run"):

    # ----------------------------
    # Model Evaluation
    # ----------------------------
    val_loss, val_acc = model.evaluate(val_gen)

    val_gen.reset()

    y_true = val_gen.classes
    y_pred_prob = model.predict(val_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # ----------------------------
    # Log Metrics
    # ----------------------------
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # ----------------------------
    # Save Confusion Matrix Artifact
    # ----------------------------
    os.makedirs("artifacts/evaluation", exist_ok=True)

    cm_path = "artifacts/evaluation/confusion_matrix.txt"
    np.savetxt(cm_path, cm, fmt="%d")

    mlflow.log_artifact(cm_path)
