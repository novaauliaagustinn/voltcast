import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import os

# ==========================
# FUNGSI UNTUK SET BACKGROUND & HEADER
# ==========================
def set_background(image_path, header_image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_bg = base64.b64encode(image_file.read()).decode()

        header_encoded = ""
        if os.path.exists(header_image_path):
            with open(header_image_path, "rb") as header_file:
                header_encoded = base64.b64encode(header_file.read()).decode()

        css_style = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_bg}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            background-color: #F4F4F4;
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 110px !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            border: none !important;
            box-shadow: none !important;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999 !important;
        }}

        header[data-testid="stHeader"]::before {{
            content: "";
            position: absolute;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            background-image: url("data:image/png;base64,{header_encoded}");
            background-repeat: no-repeat;
            background-position: top center;
            background-size: contain;
            z-index: 0;
        }}

        .block-container {{
            padding-top: 120px !important;
        }}

        html, body, .main, .block-container, h1, h2, h3, h4, h5, h6, p, label, span {{
            font-family: "Roboto", "Helvetica", "Arial", sans-serif !important;
            color: #003366 !important;
        }}
        

        /* === CARD PREDIKSI === */
        .prediction-card {{
            background-color: #f8f9fa;
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: 0.3s;
        }}
        .prediction-card:hover {{
            background-color: #e9f5ff;
            transform: translateY(-3px);
        }}
        .feature-name {{
            font-weight: 600;
            color: #333;
            font-size: 16px;
        }}
        .feature-value {{
            font-size: 18px;
            color: #007bff;
            font-weight: bold;
        }}

        /* === TOMBOL 'LIHAT HASIL PREDIKSI' === */
        .stButton > button {{
            background-color: #00BCD4 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 28px !important;
            cursor: pointer !important;
            transition: all 0.25s ease-in-out !important;
            box-shadow: 0 3px 6px rgba(0, 188, 212, 0.25) !important;
            text-shadow: none !important;
            -webkit-text-fill-color: #ffffff !important;  /* Tambahan penting */
        }}


        .stButton > button:hover {{
            background-color: #00ACC1 !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 12px rgba(0, 188, 212, 0.35) !important;
        }}

        .stButton > button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 5px rgba(0, 188, 212, 0.25) !important;
        }}

        div.stButton {{
            display: flex;
            justify-content: lefts;
            align-items: left;
            margin-top: 10px;
            margin-bottom: 25px;
        }}
        </style>
        """
        st.markdown(css_style, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Background image tidak ditemukan!")


# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Volt Cast PLN", layout="centered")
set_background("background.png", "HEADER.png")

# === Hilangkan tombol Deploy dan titik tiga Streamlit tanpa ganggu header ===
hide_menu_style = """
    <style>
    [data-testid="stToolbar"] {display: none !important;}   /* Hilangkan toolbar (Deploy & titik tiga) */
    #MainMenu {visibility: hidden !important;}              /* Hilangkan menu utama Streamlit */
    footer {visibility: hidden !important;}                 /* Sembunyikan footer Streamlit */
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


# === TAMBAHKAN BAGIAN INI ===
st.markdown("""
    <div style='text-align: left; margin-top: 20px; margin-bottom: 25px;'>
        <h1 style='font-size: 40px; color: #003366; font-weight: 700; margin-bottom: 5px;'>
            VOLT CAST: Energy Forecasting System
        </h1>
        <p style='font-size: 18px; color: #004d66;'>
            Prediksi dan analisis pergolongan KWH serta tarif listrik dengan teknologi AI berbasis 
            Machine Learning Model Multiple Random Forest Regressor
        </p>
    </div>
""", unsafe_allow_html=True)

# ==========================
# INPUT BULAN & TAHUN (Dropdown Scrollable)
# ==========================
col1, col2 = st.columns(2)

with col1:
    bulan_dict = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
        5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
        9: "September", 10: "Oktober", 11: "November", 12: "Desember"
    }
    bulan_nama = st.selectbox(
        "Pilih Bulan",
        options=list(bulan_dict.values()),
        index=0,
        help="Pilih bulan untuk prediksi."
    )
    # Konversi nama bulan ke angka
    bulan_input = list(bulan_dict.keys())[list(bulan_dict.values()).index(bulan_nama)]

with col2:
    tahun_input = st.selectbox(
        "Pilih Tahun",
        options=[tahun for tahun in range(2024, 2050)],
        index=2,
        help="Pilih tahun untuk prediksi."
    )

# ==========================
# TOMBOL PREDIKSI
# ==========================
if st.button("Lihat Hasil Prediksi"):
    # Ubah ke ordinal
    tanggal = pd.Timestamp(year=tahun_input, month=bulan_input, day=1)
    bulan_ordinal = tanggal.toordinal()

    # Baca data asli
    df = pd.read_csv("data_forecasting.csv")

    # Load model
    model = joblib.load("forecasting_regressor.pkl")
    fitur_training = model.feature_names_in_

    # Siapkan input
    X_new = pd.DataFrame(np.zeros((1, len(fitur_training))), columns=fitur_training)
    for col in fitur_training:
        if col.endswith("_prev"):
            base_col = col.replace("_prev", "")
            if base_col in df.columns:
                X_new[col] = df[base_col].iloc[-1]
    if "BULAN" in X_new.columns:
        X_new["BULAN"] = bulan_ordinal

    # Prediksi
    prediksi = model.predict(X_new)
    target_cols = df.columns.drop("BULAN")
    hasil_prediksi = pd.DataFrame(prediksi, columns=target_cols)
    hasil_prediksi = hasil_prediksi.T.reset_index().rename(columns={"index": "Fitur", 0: "Nilai"})
    hasil_prediksi["Nilai"] = hasil_prediksi["Nilai"].apply(lambda x: f"{x:,.0f}")

    # Kamus bulan Indonesia
    bulan_id = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
        5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
        9: "September", 10: "Oktober", 11: "November", 12: "Desember"
    }

    # Gunakan nama bulan Indonesia
    nama_bulan_id = bulan_id[bulan_input]

    st.markdown(f"### Prediksi untuk **{nama_bulan_id} {tahun_input}**")
    # Tampilkan hasil sebagai cards
    # === TAMPILKAN HASIL DALAM GRID 5 KOLOM DENGAN CARD SAMA UKURAN ===
    cols_per_row = 5
    rows = [hasil_prediksi.iloc[i:i+cols_per_row] for i in range(0, len(hasil_prediksi), cols_per_row)]

    # CSS tambahan agar ukuran seragam & tombol sama gaya
    st.markdown("""
    <style>
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 10px;
        margin: 8px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        text-align: left;             /* ubah dari center ke left */
        height: 120px;                /* tinggi seragam */
        width: 100%;                  /* isi penuh kolom */
        display: flex;
        flex-direction: column;
        justify-content: center;      /* tetap tengah vertikal */
        align-items: flex-start;      /* ubah dari center ke flex-start agar rata kiri */
        transition: 0.3s ease;
    }

    .prediction-card:hover {
        background-color: #e9f5ff;
        transform: translateY(-3px);
    }
    .feature-name {
        font-weight: 600;
        color: #003366;
        font-size: 15px;
        line-height: 1.2;
        margin-bottom: 5px;
    }
    .feature-value {
        font-size: 18px;
        color: #007bff;
        font-weight: bold;
    }

    /* === TOMBOL 'LIHAT HASIL PREDIKSI' & DOWNLOAD CSV (GAYA SAMA) === */
    .stButton > button,
    div.stDownloadButton > button:first-child {
        background-color: #00BCD4 !important;
        color: #ffffff !important;  /* teks putih */
        font-weight: 600 !important;
        font-size: 16px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        cursor: pointer !important;
        transition: all 0.25s ease-in-out !important;
        box-shadow: 0 3px 6px rgba(0, 188, 212, 0.25) !important;
        text-shadow: none !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .stButton > button:hover,
    div.stDownloadButton > button:first-child:hover {
        background-color: #00ACC1 !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(0, 188, 212, 0.35) !important;
    }

    .stButton > button:active,
    div.stDownloadButton > button:first-child:active {
        transform: translateY(0);
        box-shadow: 0 2px 5px rgba(0, 188, 212, 0.25) !important;
    }


    </style>
    """, unsafe_allow_html=True)

    # === GRID TAMPILAN CARD ===
    for row_chunk in rows:
        cols = st.columns(len(row_chunk))
        for col, (_, row) in zip(cols, row_chunk.iterrows()):
            with col:
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="feature-name">{row['Fitur']}</div>
                        <div class="feature-value">{row['Nilai']}</div>
                    </div>
                """, unsafe_allow_html=True)

    # === TOMBOL DOWNLOAD DENGAN GAYA SAMA ===
    hasil_csv = hasil_prediksi.copy()
    hasil_csv["Nilai"] = hasil_csv["Nilai"].replace({",": ""}, regex=True).astype(float)
    csv_data = hasil_csv.to_csv(index=False).encode("utf-8")

    # CSS fix untuk samakan gaya tombol
    st.markdown("""
    <style>
    /* Override tombol download Streamlit */
    div.stDownloadButton > button:first-child {
        background-color: #00BCD4 !important;   /* Biru toska */
        color: #002b36 !important;              /* Warna teks gelap */
        font-weight: 600 !important;
        font-size: 16px !important;
        border: none !important;
        border-radius: 10px !important;         /* Rounded */
        padding: 10px 25px !important;
        cursor: pointer !important;
        transition: all 0.25s ease-in-out !important;
        width: auto !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #00ACC1 !important;   /* Warna hover sedikit gelap */
    }
    div.stDownloadButton > button:active {
        transform: scale(0.98);
    }
    </style>
    """, unsafe_allow_html=True)

    # Tampilkan tombol download
    st.download_button(
        label="Download Hasil Prediksi (CSV)",
        data=csv_data,
        file_name=f"hasil_prediksi_{nama_bulan_id}_{tahun_input}.csv",
        mime="text/csv",
    )