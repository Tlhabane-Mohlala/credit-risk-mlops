import mlflow.sklearn
import os
import shutil

RUN_ID = "d70ccf98b40943099c3509048be13d09"
MODEL_URI = f"runs:/{RUN_ID}/model"
OUTPUT_DIR = "served_model"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

mlflow.sklearn.save_model(
    sk_model=mlflow.sklearn.load_model(MODEL_URI),
    path=OUTPUT_DIR
)

print("Model saved to served_model/")