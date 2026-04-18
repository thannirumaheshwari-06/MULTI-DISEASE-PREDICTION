import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================================================
# 📂 LOAD DATA
# ======================================================
data = pd.read_csv("heart.csv")

# ======================================================
# 🔧 ENCODE CATEGORICAL COLUMNS
# ======================================================
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# ======================================================
# 🎯 TARGET COLUMN
# ======================================================
target_col = "target" if "target" in data.columns else data.columns[-1]

X = data.drop(target_col, axis=1)
y = data[target_col]

# ======================================================
# 🔀 TRAIN-TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 🔧 SCALING (FOR KNN)
# ======================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================================
# 🤖 MODELS
# ======================================================
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="logloss",
    random_state=42
)

# ======================================================
# TRAIN MODELS
# ======================================================
knn.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ======================================================
# PREDICTIONS
# ======================================================
knn_pred = knn.predict(X_test_scaled)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# ======================================================
# 📊 METRICS FUNCTION
# ======================================================
def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ]

# ======================================================
# 📊 SIDE BY SIDE TABLE
# ======================================================
results = pd.DataFrame(
    [
        get_metrics(y_test, knn_pred),
        get_metrics(y_test, rf_pred),
        get_metrics(y_test, xgb_pred)
    ],
    columns=["Accuracy", "Precision", "Recall", "F1-score"],
    index=["KNN", "Random Forest", "XGBoost"]
)

print("\n📊 HEART DISEASE MODEL COMPARISON\n")
print(results)

# ======================================================
# 🥇 SAVE BEST MODEL (XGBOOST)
# ======================================================
with open("heart_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("\n✅ Model saved as heart_model.pkl")