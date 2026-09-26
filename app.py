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
# ==============================================================================
# 5. TAMPILAN UTAMA (MAIN PANEL)
# ==============================================================================

# ------------------------------------------------------------------------------
# A. HEADER DAN SUBHEADER
# ------------------------------------------------------------------------------

st.markdown(
    """
    <h1 style="text-align: center;">
        Sistem DSS Rekomendasi Peredam Suara Akustik
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="text-align: center; font-size: 18px;">
        Pendekatan Case-Based Reasoning (CBR) berbasis
        <b>Hibrida AHP-Entropy</b> & Standar Evaluasi
        <b>ASTM E336</b>
    </p>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# B. SUMMARY CARD REDUKSI BISING (ΔdB)
# ------------------------------------------------------------------------------

st.info(
    f"📍 **Kebutuhan Reduksi Bising ($\\Delta\\mathbf{{dB}}$):** "
    f"Intensitas Awal ({intensitas_awal:.1f} dB) − "
    f"Target ({target_suara:.1f} dB) = "
    f"**{delta_db_user:.1f} dB** | "
    f"Luas Dinding: **{luas_dinding:.1f} m²**"
)

# ------------------------------------------------------------------------------
# C. NAVIGASI MULTI-TAB
# ------------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "💡 Solusi Rekomendasi Utama",
        "📊 Analisis Kasus Mirip (Top-10)",
        "⚙️ Transparansi Algoritma (AHP & Entropy)",
    ]
)


# ==============================================================================
# TAB 1: SOLUSI REKOMENDASI UTAMA
# ==============================================================================

with tab1:

    st.subheader(
        "Rekomendasi Spesifikasi Peredam "
        "(Hasil Adaptasi CAHWM)"
    )

    # --------------------------------------------------------------------------
    # 1. EMPAT KARTU METRIK UTAMA
    # --------------------------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    # --- Material ---
    with c1:
        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                text-align: center;
            ">
                <h4>Material Utama</h4>
                <h3>{adapted_result["material_peredam"]}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Ketebalan ---
    with c2:
        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                text-align: center;
            ">
                <h4>Ketebalan Sistem</h4>
                <h3>{adapted_result["ketebalan_peredam_mm"]} mm</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Biaya ---
    with c3:
        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                text-align: center;
            ">
                <h4>Estimasi Biaya Total</h4>
                <h3>Rp {adapted_result["est_biaya_total_rp"]:,}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Confidence ---
    with c4:

        if adapted_result["tingkat_rekomendasi"] == "Tinggi":
            badge_color = "#22C55E"
        else:
            badge_color = "#F59E0B"

        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                text-align: center;
            ">
                <h4>Tingkat Kepercayaan</h4>
                <h3 style="color: {badge_color};">
                    {adapted_result["max_similarity_pct"]}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")


    # --------------------------------------------------------------------------
    # 2. PERINGATAN FISIKA AKUSTIK
    # --------------------------------------------------------------------------

    if adapted_result["flanking_rule_applied"]:

        st.warning(
            """
            **⚠️ Penyesuaian Fisika Akustik (Flanking Transmission Rule)**

            Target reduksi bising pengguna tergolong tinggi
            (**> 25 dB**).

            Pemasangan peredam secara langsung menempel pada dinding
            berpotensi mengalami kebocoran energi/getaran mekanis
            (*flanking transmission*).

            Sistem secara otomatis menyesuaikan metode menjadi:

            **Sistem Air Gap / Double Wall Structure**
            """
        )


    # --------------------------------------------------------------------------
    # 3. BREAKDOWN DETAIL KONSTRUKSI & KALKULASI ASTM E336
    # --------------------------------------------------------------------------

    col_left, col_right = st.columns([1.1, 0.9])

    # ==========================================================================
    # KOLOM KIRI
    # ==========================================================================

    with col_left:

        st.markdown("### 🏢 Detail Konstruksi & Pemasangan")

        st.markdown(
            f"""
            - **Bahan Peredam:**  
              `{adapted_result["material_peredam"]}`

            - **Metode Pemasangan:**  
              `{adapted_result["sistem_pemasangan"]}`

            - **Ketebalan Peredam:**  
              `{adapted_result["ketebalan_peredam_mm"]} mm`

            - **Estimasi Biaya per m²:**  
              `Rp {adapted_result["biaya_per_m2_rp"]:,} / m²`

            - **Status Rekomendasi:**  
              **{adapted_result["tingkat_rekomendasi"]}**
            """
        )

    # ==========================================================================
    # KOLOM KANAN
    # ==========================================================================

    with col_right:

        st.markdown("### 📐 Kalkulasi Evaluasi Akustik (ASTM E336)")

        st.markdown(
            f"""
            - **Transmission Loss (TL) Material HWM:**  
              `{adapted_result["tl_material_hwm_db"]} dB`

            - **Koreksi Geometri Ruangan:**  
              `{adapted_result["koreksi_geometri_db"]} dB`

            - **Noise Reduction (NR) Lapangan Aktual:**  
              **`{adapted_result["est_nr_aktual_db"]} dB`**

            - **Sisa Suara Terdengar di Luar:**  
              **`{adapted_result["est_suara_luar_db"]} dB`**
            """
        )


# ==============================================================================
# TAB 2: ANALISIS KASUS MIRIP (TOP-10 CANDIDATE CASES)
# ==============================================================================

with tab2:

    st.subheader(
        "10 Kasus Historis Paling Mirip "
        "(Phase 2 Retrieval)"
    )

    # --------------------------------------------------------------------------
    # 1. GRAFIK SKOR KEMIRIPAN
    # --------------------------------------------------------------------------

    fig_sim = px.bar(
        top_candidates,
        x="similarity_score",
        y="material_peredam",
        orientation="h",
        color="similarity_score",
        color_continuous_scale="Viridis",
        labels={
            "similarity_score": "Skor Kemiripan (Similarity)",
            "material_peredam": "Material Peredam",
        },
        title="Peringkat Kemiripan Kasus Historis terhadap Ruangan User",
    )

    fig_sim.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        },
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
    )

    st.plotly_chart(
        fig_sim,
        use_container_width=True,
    )


    # --------------------------------------------------------------------------
    # 2. TABEL DETAIL KANDIDAT
    # --------------------------------------------------------------------------

    st.markdown(
        "**Tabel Detail 10 Kandidat Kasus Teratas:**"
    )

    display_cols = [
        "similarity_score",
        "material_peredam",
        "ketebalan_peredam_mm",
        "sistem_pemasangan",
        "estimasi_noise_reduction_db",
        "estimasi_biaya_rp",
        "luas_ruangan_m2",
        "delta_db",
    ]

    df_show = top_candidates[display_cols].copy()

    # Format similarity menjadi persen
    df_show["similarity_score"] = (
        df_show["similarity_score"] * 100
    ).map(
        "{:.2f}%".format
    )

    # Format biaya
    df_show["estimasi_biaya_rp"] = (
        df_show["estimasi_biaya_rp"]
        .map("Rp {:,}".format)
    )

    # Rename kolom agar lebih mudah dibaca user
    df_show = df_show.rename(
        columns={
            "similarity_score": "Similarity",
            "material_peredam": "Material",
            "ketebalan_peredam_mm": "Ketebalan (mm)",
            "sistem_pemasangan": "Sistem Pemasangan",
            "estimasi_noise_reduction_db": "Estimasi NR (dB)",
            "estimasi_biaya_rp": "Estimasi Biaya",
            "luas_ruangan_m2": "Luas Ruangan (m²)",
            "delta_db": "Delta dB",
        }
    )

    st.dataframe(
        df_show,
        use_container_width=True,
        height=280,
        hide_index=True,
    )


# ==============================================================================
# TAB 3: TRANSPARANSI ALGORITMA
# ==============================================================================

with tab3:

    st.subheader(
        "Transparansi Pembobotan Kriteria "
        "(AHP vs Shannon Entropy)"
    )

    col_a, col_b = st.columns([1.2, 0.8])


    # ==========================================================================
    # KOLOM A — GRAFIK PEMBOBOTAN
    # ==========================================================================

    with col_a:

        labels = [
            "Luas Ruangan",
            "Tebal Dinding",
            "Luas Dinding",
            "Intensitas Awal",
            "Target Suara",
            "Delta dB",
        ]

        # ----------------------------------------------------------------------
        # Grafik Komparasi Bobot AHP, Entropy, dan Hybrid
        # ----------------------------------------------------------------------

        fig_w = go.Figure()

        # AHP
        fig_w.add_trace(
            go.Bar(
                x=labels,
                y=Ws,
                name="Subjektif (AHP - Ws)",
                marker_color="#3B82F6",
            )
        )

        # Shannon Entropy
        fig_w.add_trace(
            go.Bar(
                x=labels,
                y=Wo,
                name="Objektif (Entropy - Wo)",
                marker_color="#10B981",
            )
        )

        # Hybrid
        fig_w.add_trace(
            go.Bar(
                x=labels,
                y=Wh,
                name="Hibrida (SLSQP - Wh)",
                marker_color="#F59E0B",
            )
        )

        fig_w.update_layout(
            barmode="group",
            title="Perbandingan Distribusi Bobot Kriteria",
            xaxis_title="Kriteria Ruangan",
            yaxis_title="Nilai Bobot (Total = 1.0)",
            height=380,
        )

        st.plotly_chart(
            fig_w,
            use_container_width=True,
        )


    # ==========================================================================
    # KOLOM B — PARAMETER OPTIMASI
    # ==========================================================================

    with col_b:

        st.markdown("### ⚙️ Parameter Optimasi Hibrida")

        st.markdown(
            f"""
            - **Faktor Bobot Optimal ($\\alpha_{{opt}}$):**  
              `{alpha_opt:.4f}`

            - **Bobot Hibrida Akhir ($W_h$):**

            $$
            W_h = \\alpha \\cdot W_s +
            (1 - \\alpha) \\cdot W_o
            $$
            """
        )

        st.markdown(
            """
            **Penjelasan Singkat:**

            - **AHP ($W_s$):**  
              Menangkap preferensi atau tingkat kepentingan
              berdasarkan penilaian pakar akustik.

            - **Shannon Entropy ($W_o$):**  
              Menangkap informasi berdasarkan keberagaman
              dan sebaran data historis.

            - **SLSQP ($W_h$):**  
              Mengoptimasi kombinasi bobot AHP dan Entropy
              berdasarkan fungsi objektif yang digunakan sistem.
            """
        )


# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")

st.caption(
    "Sistem Pendukung Keputusan Peredam Suara Akustik • "
    "Menggunakan Algoritma CBR, HWM, & Standardisasi ASTM E336"
)