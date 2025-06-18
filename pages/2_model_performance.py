# pages/2_Model_Performance.py
import streamlit as st
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Model Performance")

# Load model dan data
model = joblib.load("model/model.pkl")
le_dict = joblib.load("model/encoders.pkl")
df = pd.read_csv("predic_tabel.csv").drop(columns=["No"], errors='ignore')

# Encode ulang
for col in df.columns:
    if col in le_dict:
        df[col] = le_dict[col].transform(df[col])

X = df.drop("Hasil", axis=1)
y = df["Hasil"]
y_pred = model.predict(X)

st.write("### Classification Report")
st.text(classification_report(y, y_pred))

st.write("### Confusion Matrix")
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(plt)
