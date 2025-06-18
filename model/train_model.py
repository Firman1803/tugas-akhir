import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("predic_tabel.csv")
df = df.drop(columns=["No"], errors="ignore")  # drop kolom No jika ada

# Label encoding
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Split data
X = df.drop("Hasil", axis=1)
y = df["Hasil"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Simpan model & encoder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(le_dict, "model/encoders.pkl")

print("Model dan encoder berhasil disimpan.")
