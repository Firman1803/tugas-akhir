# pages/3_Prediction.py
import streamlit as st
import joblib

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
