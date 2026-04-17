import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. DIABETES DATASET
# =========================
diabetes_df = pd.read_csv("diabetes.csv")

X_d = diabetes_df.drop("Outcome", axis=1)
y_d = diabetes_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X_d, y_d,
    test_size=0.2,
    random_state=42,
    stratify=y_d
)

diabetes_model = XGBClassifier(eval_metric="logloss")
diabetes_model.fit(X_train, y_train)

diabetes_pred = diabetes_model.predict(X_test)
print("Diabetes Accuracy:", accuracy_score(y_test, diabetes_pred))


# =========================
# 2. HEART DATASET (FIXED CATEGORICAL ERROR)
# =========================
heart_df = pd.read_csv("heart.csv")

# Convert target column
if "num" in heart_df.columns:
    heart_df["target"] = heart_df["num"].apply(lambda x: 0 if x == 0 else 1)
    heart_df.drop(columns=["num"], inplace=True)

X_h = heart_df.drop("target", axis=1)
y_h = heart_df["target"]

# 🔥 FIX: Encode categorical columns
for col in X_h.columns:
    if X_h[col].dtype == "object":
        le = LabelEncoder()
        X_h[col] = le.fit_transform(X_h[col])

X_train, X_test, y_train, y_test = train_test_split(
    X_h, y_h,
    test_size=0.2,
    random_state=42,
    stratify=y_h
)

heart_model = XGBClassifier(eval_metric="logloss")
heart_model.fit(X_train, y_train)

heart_pred = heart_model.predict(X_test)
print("Heart Accuracy:", accuracy_score(y_test, heart_pred))


# =========================
# 3. STORE MODELS IN DICTIONARY
# =========================
models = {
    "diabetes": diabetes_model,
    "heart": heart_model
}

# =========================
# 4. SAVE MODELS USING PICKLE
# =========================
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(models["diabetes"], f)

with open("heart_model.pkl", "wb") as f:
    pickle.dump(models["heart"], f)

print("\n✅ Models trained and saved successfully!")