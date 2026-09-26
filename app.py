
import cbr_engine
import pandas as pd
import streamlit as st


# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="Sistem Pakar Peredam Suara (CBR)",
    page_icon="🔊",
    layout="wide",
)


# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown(
    """
    <style>
    /* =========================================================
       CARD UTAMA
       ========================================================= */

    .spec-card {
        background-color: #f5f7fa;
        border: 1px solid #d9dee7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        min-height: 140px;
    }

    .spec-title {
        font-size: 16px;
        font-weight: 700;
        color: #333333;
        margin-bottom: 10px;
    }

    .spec-value {
        font-size: 22px;
        font-weight: 700;
        color: #111111;
        margin-bottom: 8px;
    }

    .spec-description {
        font-size: 14px;
        color: #555555;
    }

    /* =========================================================
       CARD ANALISIS
       ========================================================= */

    .analysis-card {
        background-color: #f8f9fb;
        border: 1px solid #dfe3e8;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        min-height: 110px;
    }

    .analysis-title {
        font-size: 14px;
        font-weight: 600;
        color: #555555;
        margin-bottom: 8px;
    }

    .analysis-value {
        font-size: 24px;
        font-weight: 700;
        color: #111111;
    }

    /* =========================================================
       SIDEBAR
       ========================================================= */

    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    /* =========================================================
       DARK MODE
       ========================================================= */

    @media (prefers-color-scheme: dark) {

        .spec-card {
            background-color: #262730;
            border-color: #444750;
        }

        .spec-title {
            color: #dddddd;
        }

        .spec-value {
            color: #ffffff;
        }

        .spec-description {
            color: #bbbbbb;
        }

        .analysis-card {
            background-color: #262730;
            border-color: #444750;
        }

        .analysis-title {
            color: #bbbbbb;
        }

        .analysis-value {
            color: #ffffff;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# LOAD ENGINE & DATASET
# ==============================================================================

@st.cache_data
def get_initialized_engine():
    csv_file = "peredam_suara_clean.csv"

    return cbr_engine.load_and_initialize_engine(csv_file)


try:
    engine_data = get_initialized_engine()

    df_cases = engine_data["df_cases"]
    Wh = engine_data["Wh"]
    v_min = engine_data["v_min"]
    v_max = engine_data["v_max"]

except Exception as e:
    st.error(f"Gagal memuat dataset atau engine CBR: {e}")
    st.stop()


# ==============================================================================
# SIDEBAR — FORM INPUT USER
# ==============================================================================

st.sidebar.header("📝 Input Parameter Ruangan")

st.sidebar.markdown(
    "Masukkan spesifikasi ruangan dan kebutuhan akustik Anda di bawah ini."
)


with st.sidebar.form(key="user_input_form"):

    luas_ruangan_m2 = st.number_input(
        "Luas Ruangan (m²)",
        min_value=5.0,
        max_value=200.0,
        value=18.0,
        step=1.0,
    )

    ketebalan_dinding_cm = st.number_input(
        "Ketebalan Dinding (cm)",
        min_value=5.0,
        max_value=50.0,
        value=12.0,
        step=1.0,
    )

    luas_dinding_m2 = st.number_input(
        "Luas Dinding yang Diberi Peredam (m²)",
        min_value=5.0,
        max_value=200.0,
        value=30.0,
        step=1.0,
    )

    intensitas_suara_awal_db = st.slider(
        "Intensitas Suara Bising Awal (dB)",
        min_value=40.0,
        max_value=120.0,
        value=85.0,
        step=1.0,
    )

    target_suara_db = st.slider(
        "Target Suara yang Diinginkan (dB)",
        min_value=20.0,
        max_value=60.0,
        value=40.0,
        step=1.0,
    )

    top_k_val = st.slider(
        "Jumlah Kasus Mirip (Top-K)",
        min_value=3,
        max_value=15,
        value=10,
        step=1,
    )

    submit_button = st.form_submit_button(
        label="🔍 Hitung Rekomendasi",
        use_container_width=True,
    )


# ==============================================================================
# HALAMAN UTAMA
# ==============================================================================

st.title("🔊 Sistem Pakar Rekomendasi Peredam Suara")

st.markdown(
    """
    Sistem berbasis **Case-Based Reasoning (CBR)** dengan bobot hibrida
    **AHP + Shannon Entropy + SLSQP** dan **Grey Relational Analysis (GRA)**.
    """
)

st.divider()


# ==============================================================================
# USER INPUT
# ==============================================================================

delta_db_calc = (
    intensitas_suara_awal_db - target_suara_db
)

user_input = {
    "luas_ruangan_m2": luas_ruangan_m2,
    "ketebalan_dinding_cm": ketebalan_dinding_cm,
    "luas_dinding_m2": luas_dinding_m2,
    "intensitas_suara_awal_db": intensitas_suara_awal_db,
    "target_suara_db": target_suara_db,
    "delta_db": delta_db_calc,
}


# ==============================================================================
# RETRIEVE
# ==============================================================================

try:

    top_k_candidates = cbr_engine.retrieve_top_k(
        df_cases,
        user_input,
        Wh,
        v_min,
        v_max,
        K=top_k_val,
    )

except Exception as e:

    st.error(f"Gagal melakukan proses retrieval CBR: {e}")
    st.stop()


# ==============================================================================
# ADAPT
# ==============================================================================

try:

    adapted_result = cbr_engine.adapt_solution(
        top_k_candidates,
        user_input,
        Wh,
    )

except Exception as e:

    st.error(f"Gagal melakukan proses adaptasi solusi: {e}")
    st.stop()


# ==============================================================================
# SECTION 1 — RINGKASAN REKOMENDASI UTAMA
# ==============================================================================

st.subheader("🎯 Rekomendasi Solusi Peredam Suara")


# ------------------------------------------------------------------------------
# FLANKING RULE
# ------------------------------------------------------------------------------

if adapted_result["flanking_rule_applied"]:

    st.warning(
        "⚠️ **Catatan Flanking Transmission:** "
        "Kebutuhan reduksi suara tinggi (>25 dB). "
        "Sistem pemasangan disesuaikan menggunakan rongga udara "
        "(air gap) untuk membantu mengurangi transmisi getaran."
    )


# ------------------------------------------------------------------------------
# KARTU REKOMENDASI
# ------------------------------------------------------------------------------

col1, col2 = st.columns(2)


# ==============================================================================
# KOLOM KIRI
# ==============================================================================

with col1:

    # --------------------------------------------------------------------------
    # MATERIAL
    # --------------------------------------------------------------------------

    with st.container(border=True):

        st.markdown("### 🧱 Material Peredam Utama")

        st.markdown(
            f"## {adapted_result['material_peredam']}"
        )

        st.markdown(
            f"Ketebalan rekomendasi: "
            f"**{adapted_result['ketebalan_peredam_mm']} mm**"
        )


    # --------------------------------------------------------------------------
    # SISTEM PEMASANGAN
    # --------------------------------------------------------------------------

    with st.container(border=True):

        st.markdown("### 🔧 Sistem Pemasangan")

        st.markdown(
            f"**{adapted_result['sistem_pemasangan']}**"
        )

        st.caption(
            "Sistem pemasangan yang direkomendasikan berdasarkan "
            "hasil adaptasi CBR."
        )


# ==============================================================================
# KOLOM KANAN
# ==============================================================================

with col2:

    # --------------------------------------------------------------------------
    # BIAYA
    # --------------------------------------------------------------------------

    with st.container(border=True):

        st.markdown("### 💰 Estimasi Total Biaya")

        st.markdown(
            f"## Rp {adapted_result['est_biaya_total_rp']:,}"
        )

        st.markdown(
            f"Estimasi biaya per m²: "
            f"**Rp {adapted_result['biaya_per_m2_rp']:,} / m²**"
        )


    # --------------------------------------------------------------------------
    # KEMIRIPAN
    # --------------------------------------------------------------------------

    with st.container(border=True):

        st.markdown("### 🎯 Tingkat Kesesuaian Kasus")

        st.markdown(
            f"## {adapted_result['max_similarity_pct']}%"
        )

        st.markdown(
            f"Tingkat rekomendasi: "
            f"**{adapted_result['tingkat_rekomendasi']}**"
        )


st.divider()


# ==============================================================================
# SECTION 2 — DETAIL PERHITUNGAN AKUSTIK
# ==============================================================================

st.subheader("📊 Detail Perhitungan Akustik")


mcol1, mcol2, mcol3, mcol4 = st.columns(4)


# ==============================================================================
# TARGET REDUKSI
# ==============================================================================

with mcol1:

    with st.container(border=True):

        st.markdown("**Target Reduksi (Δ dB)**")

        st.markdown(
            f"### {user_input['delta_db']:.1f} dB"
        )


# ==============================================================================
# REDUKSI MATERIAL
# ==============================================================================

with mcol2:

    with st.container(border=True):

        st.markdown("**Reduksi Material (TL HWM)**")

        st.markdown(
            f"### {adapted_result['tl_material_hwm_db']} dB"
        )


# ==============================================================================
# KOREKSI GEOMETRI
# ==============================================================================

with mcol3:

    with st.container(border=True):

        st.markdown("**Koreksi Geometri Ruang**")

        st.markdown(
            f"### {adapted_result['koreksi_geometri_db']} dB"
        )


# ==============================================================================
# ESTIMASI KEBISINGAN AKHIR
# ==============================================================================

with mcol4:

    with st.container(border=True):

        st.markdown("**Est. Kebisingan Akhir**")

        st.markdown(
            f"### {adapted_result['est_suara_luar_db']} dB"
        )


st.divider()


# ==============================================================================
# SECTION 3 — TOP-K KASUS SERUPA
# ==============================================================================

st.subheader(
    f"📋 Top-{top_k_val} Kasus Basis Data yang Paling Mirip"
)


# ------------------------------------------------------------------------------
# COPY DATAFRAME
# ------------------------------------------------------------------------------

display_df = top_k_candidates.copy()


# ------------------------------------------------------------------------------
# FORMAT SIMILARITY
# ------------------------------------------------------------------------------

display_df["similarity_score"] = (
    display_df["similarity_score"] * 100
).round(2).astype(str) + " %"


# ------------------------------------------------------------------------------
# FORMAT BIAYA
# ------------------------------------------------------------------------------

if "estimasi_biaya_rp" in display_df.columns:

    display_df["estimasi_biaya_rp"] = display_df[
        "estimasi_biaya_rp"
    ].apply(
        lambda x: f"Rp {x:,.0f}"
        if pd.notna(x)
        else "-"
    )


# ------------------------------------------------------------------------------
# RENAME KOLOM
# ------------------------------------------------------------------------------

display_df = display_df.rename(
    columns={
        "id_kasus": "ID Kasus",
        "luas_ruangan_m2": "Luas R. (m²)",
        "ketebalan_dinding_cm": "Tebal Dinding (cm)",
        "luas_dinding_m2": "Luas Dinding (m²)",
        "intensitas_suara_awal_db": "Suara Awal (dB)",
        "target_suara_db": "Target (dB)",
        "material_peredam": "Material",
        "sistem_pemasangan": "Sistem Pemasangan",
        "ketebalan_peredam_mm": "Tebal Peredam (mm)",
        "estimasi_biaya_rp": "Biaya (Rp)",
        "similarity_score": "Kemiripan (%)",
    }
)


# ------------------------------------------------------------------------------
# TABEL
# ------------------------------------------------------------------------------

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()

st.caption(
    "Sistem Pakar Peredam Suara — Case-Based Reasoning (CBR)"
)
