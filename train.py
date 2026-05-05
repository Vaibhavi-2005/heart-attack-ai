import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 📊 Load dataset
df = pd.read_csv("heart_data.csv")

print("✅ Dataset Loaded Successfully!")
print(df.head())
print(df.columns)
# 🔍 Features (must match app.py)
X = df[["age", "chol", "trestbps"]]
y = df["target"]
print(df["target"].value_counts())
# ⚖️ Handle imbalance (IMPORTANT)
print("\nTarget Distribution:")
print(y.value_counts())

# ✂️ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 📏 Scale data (improves model stability)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Train model (improved)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight={0:1, 1:1},  # try removing imbalance bias first
    random_state=42
)

model.fit(X_train, y_train)

# 📈 Predictions
y_pred = model.predict(X_test)

# 📊 Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# 📋 Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Save BOTH model + scaler (IMPORTANT)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved as model.pkl")
print("✅ Scaler saved as scaler.pkl")