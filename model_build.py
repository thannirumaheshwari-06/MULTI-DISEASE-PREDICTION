import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import pickle
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("finaldataset.csv")

# Remove Healthy samples
df = df[df['Disease'] != 'Healthy']

# Features and Target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Encode disease labels
le = LabelEncoder()
y = le.fit_transform(y)

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 3. Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# -----------------------------
# 4. Support Vector Machine
# -----------------------------
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# -----------------------------
# 5. XGBoost
# -----------------------------
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# -----------------------------
# 6. Calculate Metrics
# -----------------------------
metrics_data = {
    "Model": ["Logistic Regression", "SVM", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_xgb)
    ],
    "Precision": [
        precision_score(y_test, y_pred_log, average='weighted'),
        precision_score(y_test, y_pred_svm, average='weighted'),
        precision_score(y_test, y_pred_xgb, average='weighted')
    ],
    "Recall": [
        recall_score(y_test, y_pred_log, average='weighted'),
        recall_score(y_test, y_pred_svm, average='weighted'),
        recall_score(y_test, y_pred_xgb, average='weighted')
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_log, average='weighted'),
        f1_score(y_test, y_pred_svm, average='weighted'),
        f1_score(y_test, y_pred_xgb, average='weighted')
    ]
}

results = pd.DataFrame(metrics_data)

print("\nModel Performance Metrics:\n")
print(results)

# -----------------------------
# 7. Best Model
# -----------------------------
best_idx = results['Accuracy'].idxmax()
best_model = results.loc[best_idx, 'Model']
best_accuracy = results.loc[best_idx, 'Accuracy']
best_precision = results.loc[best_idx, 'Precision']
best_recall = results.loc[best_idx, 'Recall']
best_f1 = results.loc[best_idx, 'F1 Score']

print("\nBest Model for Disease Prediction:")
print(f"Model: {best_model}")
print(f"Accuracy : {best_accuracy:.3f}")
print(f"Precision: {best_precision:.3f}")
print(f"Recall   : {best_recall:.3f}")
print(f"F1 Score : {best_f1:.3f}")

# -----------------------------
# 8. Save only XGBoost model, scaler, and label encoder
# -----------------------------
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nXGBoost model, scaler, and label encoder have been saved as pickle files.")

# -----------------------------
# 9. Visualization
# -----------------------------
results.set_index("Model").plot(
    kind='bar',
    figsize=(10,6),
    colormap='Set2'
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()

