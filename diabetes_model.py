import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# 📂 Load dataset
# -------------------------------
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# -------------------------------
# 🔀 Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🔧 Scaling (for KNN only)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 🤖 Models
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="logloss",
    random_state=42
)

# -------------------------------
# Training
# -------------------------------
knn.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
knn_pred = knn.predict(X_test_scaled)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# -------------------------------
# 📊 Metric function
# -------------------------------
def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    ]

# -------------------------------
# 📊 Comparison table
# -------------------------------
results = pd.DataFrame(
    [
        get_metrics(y_test, knn_pred),
        get_metrics(y_test, rf_pred),
        get_metrics(y_test, xgb_pred)
    ],
    columns=["Accuracy", "Precision", "Recall", "F1-score"],
    index=["KNN", "Random Forest", "XGBoost"]
)

print("\n📊 Model Comparison:\n")
print(results)

# -------------------------------
# 🥇 Select best model (XGBoost here)
# -------------------------------
best_model = xgb

# -------------------------------
# 💾 Save model as pickle file
# -------------------------------
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\n✅ Model saved successfully as diabetes_xgboost_model.pkl")