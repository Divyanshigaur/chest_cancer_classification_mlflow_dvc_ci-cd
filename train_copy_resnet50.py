import os
import yaml
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
import shutil

# ----------------------------
# Load parameters
# ----------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

EPOCHS = params["training"]["epochs"]
BATCH_SIZE = params["training"]["batch_size"]
IMG_SIZE = params["data"]["img_size"]

# IMPORTANT: lower LR for pretrained model
LR = 1e-4
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# ----------------------------
# Sanity check: NO DATA LEAKAGE
# ----------------------------
def get_all_filenames(dir_path):
    files = set()
    for root, _, filenames in os.walk(dir_path):
        for f in filenames:
            files.add(f)
    return files

train_files = get_all_filenames(TRAIN_DIR)
val_files = get_all_filenames(VAL_DIR)

overlap = train_files.intersection(val_files)

assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping files: {overlap}"
print("✅ No data leakage detected between train and val")


TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# ----------------------------
# Data generators (CORRECT WAY)
# ----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ----------------------------
# Model definition (ResNet50)
# ----------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# Callbacks (VERY IMPORTANT)
# ----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# ----------------------------
# MLflow tracking
# ----------------------------
mlflow.set_experiment("Chest_Cancer_Detection")

run_name = f"ResNet50_TransferLearning_Epoch{EPOCHS}"

with mlflow.start_run(run_name=run_name):

    mlflow.log_param("model", "ResNet50")
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)

    shutil.copy("src/train_resnet50.py", "train_copy_resnet50.py")
    mlflow.log_artifact("train_copy_resnet50.py")

    # Train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr]
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen)

    val_gen.reset()
    y_true = val_gen.classes
    y_pred_prob = model.predict(val_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    np.savetxt("confusion_matrix_resnet50.txt", cm, fmt="%d")
    mlflow.log_artifact("confusion_matrix_resnet50.txt")

        # ----------------------------
    # TRUE UNSEEN TEST EVALUATION
    # ----------------------------
    TEST_DIR = "data/test"

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    test_loss, test_acc = model.evaluate(test_gen)

    test_gen.reset()
    y_test_true = test_gen.classes
    y_test_prob = model.predict(test_gen)
    y_test_pred = (y_test_prob > 0.5).astype(int).ravel()

    test_precision = precision_score(y_test_true, y_test_pred)
    test_recall = recall_score(y_test_true, y_test_pred)
    test_cm = confusion_matrix(y_test_true, y_test_pred)

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)

    np.savetxt("confusion_matrix_test.txt", test_cm, fmt="%d")
    mlflow.log_artifact("confusion_matrix_test.txt")

    os.makedirs("models", exist_ok=True)
    model.save("models/resnet50_model.h5")
    mlflow.keras.log_model(model, "model")
