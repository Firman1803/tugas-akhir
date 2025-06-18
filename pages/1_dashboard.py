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
