# app.py
import streamlit as st

st.set_page_config(page_title="Prediksi Kesehatan", layout="wide")

st.title("Aplikasi Prediksi Kesehatan")
st.markdown("""
Selamat datang di aplikasi prediksi kesehatan. Gunakan menu di sebelah kiri untuk navigasi:
- **Dashboard**: Eksplorasi Data
- **Model Performance**: Evaluasi Model
- **Prediction**: Coba Prediksi
""")

# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Dashboard - Exploratory Data Analysis")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("predic_tabel.csv")

df = load_data()
st.write("### Data Sampel")
st.dataframe(df.head())

st.write("### Distribusi Usia")
sns.countplot(x="Usia", data=df)
st.pyplot(plt)

st.write("### Distribusi Hasil")
sns.countplot(x="Hasil", data=df)
st.pyplot(plt)

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
df = pd.read_csv("predic_tabel.csv").drop(columns=["No"])
for col in df.columns:
    if df[col].dtype == 'object':
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

# pages/3_Prediction.py
import streamlit as st
import joblib
import numpy as np

st.title("Prediksi Kesehatan Pengguna")

model = joblib.load("model/model.pkl")
le_dict = joblib.load("model/encoders.pkl")

# Form input
usia = st.selectbox("Usia", le_dict['Usia'].classes_)
jenis_kelamin = st.selectbox("Jenis Kelamin", le_dict['Jenis_Kelamin'].classes_)
merokok = st.selectbox("Merokok", le_dict['Merokok'].classes_)
bekerja = st.selectbox("Bekerja", le_dict['Bekerja'].classes_)
rumah_tangga = st.selectbox("Rumah Tangga", le_dict['Rumah_Tangga'].classes_)
begadang = st.selectbox("Aktivitas Begadang", le_dict['Aktivitas_Begadang'].classes_)
olahraga = st.selectbox("Aktivitas Olahraga", le_dict['Aktivitas_Olahraga'].classes_)
asuransi = st.selectbox("Asuransi", le_dict['Asuransi'].classes_)
penyakit = st.selectbox("Penyakit Bawaan", le_dict['Penyakit_Bawaan'].classes_)

if st.button("Prediksi"):
    data = [
        le_dict['Usia'].transform([usia])[0],
        le_dict['Jenis_Kelamin'].transform([jenis_kelamin])[0],
        le_dict['Merokok'].transform([merokok])[0],
        le_dict['Bekerja'].transform([bekerja])[0],
        le_dict['Rumah_Tangga'].transform([rumah_tangga])[0],
        le_dict['Aktivitas_Begadang'].transform([begadang])[0],
        le_dict['Aktivitas_Olahraga'].transform([olahraga])[0],
        le_dict['Asuransi'].transform([asuransi])[0],
        le_dict['Penyakit_Bawaan'].transform([penyakit])[0],
    ]
    pred = model.predict([data])[0]
    hasil = le_dict['Hasil'].inverse_transform([pred])[0]
    st.success(f"Prediksi Hasil: {hasil}")
