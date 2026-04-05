import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    brier_score_loss,
)
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/loan_data.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())
print("\nColumns:")
print(df.columns)

# Drop missing values for baseline model
df = df.dropna()
print("\nShape after dropping missing values:", df.shape)

# Target column
y = df["loan_status"]

# Use numeric features only
X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["loan_status"], errors="ignore")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
print(classification_report(y_test, y_pred_log))

# -----------------------------
# 3. Random Forest
# -----------------------------
rf_plain = RandomForestClassifier(n_estimators=100, random_state=42)
rf_plain.fit(X_train, y_train)

y_pred_rf_plain = rf_plain.predict(X_test)
y_prob_rf_plain = rf_plain.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_plain))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf_plain))
print(classification_report(y_test, y_pred_rf_plain))

# -----------------------------
# 4. SMOTE + Random Forest
# -----------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)

y_pred_rf_smote = rf_smote.predict(X_test)
y_prob_rf_smote = rf_smote.predict_proba(X_test)[:, 1]

print("\nRandom Forest with SMOTE")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_smote))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf_smote))
print(classification_report(y_test, y_pred_rf_smote))

# -----------------------------
# 5. Calibrated Random Forest with SMOTE
# -----------------------------
calibrated_rf = CalibratedClassifierCV(rf_smote, method="sigmoid", cv=5)
calibrated_rf.fit(X_train_resampled, y_train_resampled)

y_pred_cal = calibrated_rf.predict(X_test)
y_prob_cal = calibrated_rf.predict_proba(X_test)[:, 1]

print("\nCalibrated Random Forest with SMOTE")
print("Accuracy:", accuracy_score(y_test, y_pred_cal))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_cal))
print("Brier Score:", brier_score_loss(y_test, y_prob_cal))
print(classification_report(y_test, y_pred_cal))

# -----------------------------
# 6. Probability comparison before vs after calibration
# -----------------------------
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Before_Calibration": y_prob_rf_smote,
    "After_Calibration": y_prob_cal
})

print("\nProbability Comparison:")
print(results_df.head(10))

results_df["Difference"] = (
    results_df["Before_Calibration"] - results_df["After_Calibration"]
).abs()

results_df_sorted = results_df.sort_values(by="Difference", ascending=False)

print("\nBiggest probability adjustments:")
print(results_df_sorted.head(10))

# Save probability comparison to CSV
results_df_sorted.to_csv("probability_comparison.csv", index=False)
print("Probability comparison saved as probability_comparison.csv")

# -----------------------------
# 7. Histogram comparison
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(y_prob_rf_smote, bins=30, alpha=0.5, label="Before Calibration")
plt.hist(y_prob_cal, bins=30, alpha=0.5, label="After Calibration")

plt.legend()
plt.title("Probability Distribution Comparison")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("probability_distribution.png", bbox_inches="tight")
plt.close()

print("Histogram saved as probability_distribution.png")
# -----------------------------
# 8. SHAP Explainability
# -----------------------------
try:
    import shap

    print("Starting SHAP...")

    sample_X_test = X_test.sample(min(200, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(rf_smote)
    shap_values = explainer.shap_values(sample_X_test)

    print("Type of shap_values:", type(shap_values))

    if isinstance(shap_values, list):
        print("Length of shap_values list:", len(shap_values))
        print("Shape of shap_values[0]:", shap_values[0].shape)
        print("Shape of shap_values[1]:", shap_values[1].shape)
        shap_plot_values = shap_values[1]
    else:
        print("Shape of shap_values:", shap_values.shape)
        shap_plot_values = shap_values

    print("Shape of sample_X_test:", sample_X_test.shape)

    plt.figure()
    shap.summary_plot(shap_plot_values, sample_X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches="tight")
    plt.close()

    print("SHAP summary plot saved as shap_summary.png")

except Exception as e:
    print("SHAP step failed:", e)

# -----------------------------
# 9. MLflow Tracking
# -----------------------------
with mlflow.start_run(run_name="calibrated_rf_smote"):

    acc = accuracy_score(y_test, y_pred_cal)
    roc = roc_auc_score(y_test, y_prob_cal)
    brier = brier_score_loss(y_test, y_prob_cal)

    mlflow.log_param("model_type", "Calibrated Random Forest")
    mlflow.log_param("base_model", "RandomForest")
    mlflow.log_param("smote", True)
    mlflow.log_param("calibration_method", "sigmoid")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("brier_score", brier)

    mlflow.sklearn.log_model(calibrated_rf, "model")

    print("MLflow logging complete")