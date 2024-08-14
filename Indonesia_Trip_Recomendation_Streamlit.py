import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Judul aplikasi
st.title("Sistem Rekomendasi Trip di Indonesia")
st.sidebar.title("Filter Rekomendasi")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    tourism_with_id = pd.read_csv('Assets/tourism_with_id.csv')
    user_data = pd.read_csv('Assets/user.csv')
    tourism_rating = pd.read_csv('Assets/tourism_rating.csv')
    package_tourism = pd.read_csv('Assets/package_tourism.csv')
    return tourism_with_id, user_data, tourism_rating, package_tourism

# Panggil fungsi untuk memuat data
tourism_with_id, user_data, tourism_rating, package_tourism = load_data()

# Menampilkan data mentah (opsional)
if st.sidebar.checkbox("Tampilkan data mentah"):
    st.subheader('Data Tempat Wisata')
    st.write(tourism_with_id.head())

# Logika Rekomendasi Berbasis Konten
def content_based_recommendation(tourism_data, user_input):
    # Contoh: Menggunakan cosine similarity pada beberapa fitur untuk rekomendasi
    cosine_sim = cosine_similarity(tourism_data[['feature1', 'feature2']])
    rekomendasi = tourism_data.iloc[cosine_sim.argsort()[:, -5:]]
    return rekomendasi

# Logika Rekomendasi Berbasis Kolaboratif
def collaborative_filtering_recommendation(user_data, tourism_ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(tourism_ratings[['user_id', 'tourism_id', 'rating']], reader)
    algo = SVD()
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return algo

# Input dari pengguna
user_input = st.sidebar.text_input("Masukkan preferensi Anda")

# Tombol untuk menghasilkan rekomendasi
if st.sidebar.button("Rekomendasikan"):
    content_recommendations = content_based_recommendation(tourism_with_id, user_input)
    st.subheader('Rekomendasi Berdasarkan Konten')
    st.write(content_recommendations)

    # Placeholder untuk Rekomendasi Kolaboratif
    st.subheader('Rekomendasi Berdasarkan Kolaborasi')
    # collab_recommendations = collaborative_filtering_recommendation(user_data, tourism_rating)
    st.write("Rekomendasi Kolaboratif Akan Ditampilkan di Sini (belum diimplementasikan)")

# Jalankan aplikasi
if __name__ == "__main__":
    st.set_page_config(page_title="Sistem Rekomendasi Trip di Indonesia", layout="wide")
