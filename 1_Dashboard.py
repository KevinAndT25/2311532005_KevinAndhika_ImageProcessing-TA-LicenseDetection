import streamlit as st

st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 License Plate Detection System")

st.markdown("""
Selamat datang pada aplikasi berbasis web deteksi plat nomor kendaraan 
berbasis model YOLOv8.
""")

st.header("👨‍💻 Identitas Pengembang")

col1, col2 = st.columns([1,2])

with col1:
    st.image(
        "logo.png",
        width=250
    )

with col2:
    st.subheader("License Plate Detection")

    st.write("""
    Nama Project:
    License Plate Detection menggunakan YOLOv8
             
    Nama Mahasiswa:
    Kevin Andhika | 2311532005
        
    Framework:
    Streamlit
             
    """)

st.markdown("""
### Deskripsi Sistem
Sistem License Plate Detection adalah aplikasi berbasis web yang menggunakan model YOLOv8 untuk 
mendeteksi plat nomor kendaraan dari gambar atau kamera. Sistem menampilkan hasil deteksi 
beserta informasi seperti confidence score, jumlah objek terdeteksi, dan waktu inferensi 
melalui antarmuka yang sederhana dan mudah digunakan.
""")