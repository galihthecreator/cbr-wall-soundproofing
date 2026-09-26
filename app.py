# ==============================================================================
# STREAMLIT UI/UX DASHBOARD (app.py)
# Sistem Rekomendasi Peredam Suara Akustik (CBR - ASTM E336)
# ==============================================================================

import os
import cbr_engine
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CUSTOM CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title='Sistem Rekomendasi Peredam Suara (CBR - ASTM E336)',
    page_icon='🔊',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """

""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 2. LOAD & INITIALIZE ENGINE (WITH STREAMLIT CACHING)
# ------------------------------------------------------------------------------
@st.cache_data
def get_initialized_engine():
  csv_file = 'dataset_peredam_suara_clean.csv'
  if not os.path.exists(csv_file):
    st.error(f"File dataset '{csv_file}' tidak ditemukan di direktori lokal!")
    st.stop()
  return cbr_engine.load_and_initialize_engine(csv_file)


engine_data = get_initialized_engine()
df_cases = engine_data['df_cases']
Wh = engine_data['Wh']
Ws = engine_data['Ws']
Wo = engine_data['Wo']
alpha_opt = engine_data['alpha_opt']
v_min = engine_data['v_min']
v_max = engine_data['v_max']

# ------------------------------------------------------------------------------
# 3. SIDEBAR: FORM INPUT PARAMETER RUANGAN
# ------------------------------------------------------------------------------
st.sidebar.image(
    'https://img.icons8.com/isometric-folders/100/sound-waves.png', width=70
)
st.sidebar.title('Parameter Ruangan')
st.sidebar.caption('Masukkan kondisi ruangan & karakteristik kebisingan:')

with st.sidebar.form('user_input_form'):
  st.subheader('📐 Spesifikasi Dimensi')
  luas_ruangan = st.number_input(
      'Luas Ruangan (m²)',
      min_value=5.0,
      max_value=300.0,
      value=20.0,
      step=1.0,
  )
  luas_dinding = st.number_input(
      'Luas Dinding Pembatas (m²)',
      min_value=4.0,
      max_value=200.0,
      value=12.0,
      step=1.0,
  )
  ketebalan_dinding = st.number_input(
      'Ketebalan Dinding Eksisting (cm)',
      min_value=5.0,
      max_value=50.0,
      value=15.0,
      step=0.5,
  )
  jenis_dinding = st.selectbox(
      'Jenis Dinding Eksisting',
      [
          'Bata Merah (Plester)',
          'Hebel / Bata Ringan',
          'Beton Bertulang',
          'Gypsum Board',
          'Batako',
      ],
  )

  st.subheader('🔊 Karakteristik Bising')
  intensitas_awal = st.slider(
      'Intensitas Suara Awal (dB)',
      min_value=40.0,
      max_value=120.0,
      value=85.0,
      step=1.0,
      help='Contoh: Percakapan (60dB), Musik/Studio (85dB), Studio Band (100dB)',
  )
  target_suara = st.slider(
      'Target Suara Ruang Penerima (dB)',
      min_value=20.0,
      max_value=60.0,
      value=40.0,
      step=1.0,
      help='Contoh: Kamar Tidur (30-35dB), Kantor Hening (40dB)',
  )
  sumber_suara = st.selectbox(
      'Sumber Kebisingan',
      [
          'Lalu Lintas / Kendaraan',
          'Musik & Instrument Studio',
          'Percakapan / Suara Manusia',
          'Mesin / Peralatan Industri',
      ],
  )

  submit_btn = st.form_submit_button(
      '🔍 Hitung & Cari Rekomendasi', use_container_width=True
  )

# Kalkulasi ΔdB otomatis
delta_db_user = max(0.0, intensitas_awal - target_suara)

if target_suara >= intensitas_awal:
  st.sidebar.warning(
      '⚠️ Target suara harus lebih kecil dari intensitas suara awal.'
  )

# ------------------------------------------------------------------------------
# 4. LOGIKA EKSEKUSI PIPELINE (RETRIEVAL & ADAPTATION)
# ------------------------------------------------------------------------------
user_input_dict = {
    'luas_ruangan_m2': luas_ruangan,
    'ketebalan_dinding_cm': ketebalan_dinding,
    'luas_dinding_m2': luas_dinding,
    'intensitas_suara_awal_db': intensitas_awal,
    'target_suara_db': target_suara,
    'delta_db': delta_db_user,
    'jenis_dinding': jenis_dinding,
    'sumber_suara': sumber_suara,
}

# Jalankan kalkulasi jika tombol ditekan atau pada render awal
if submit_btn or 'top_candidates' not in st.session_state:
  top_cand = cbr_engine.retrieve_top_k(
      df_cases, user_input_dict, Wh, v_min, v_max, K=10
  )
  adapted_res = cbr_engine.adapt_solution(top_cand, user_input_dict, Wh)

  st.session_state['top_candidates'] = top_cand
  st.session_state['adapted_result'] = adapted_res
  st.session_state['last_input'] = user_input_dict

top_candidates = st.session_state['top_candidates']
adapted_result = st.session_state['adapted_result']

# ------------------------------------------------------------------------------
# 5. TAMPILAN UTAMA (MAIN PANEL)
# ------------------------------------------------------------------------------
st.markdown(
    '