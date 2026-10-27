import cbr_engine
import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="Sistem Pakar Peredam Suara Ruangan (ICBR)",
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
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        min-height: 140px;
    }

    .spec-title {
        font-size: 15px;
        font-weight: 700;
        color: #475569;
        margin-bottom: 8px;
    }

    .spec-value {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }

    .spec-description {
        font-size: 13px;
        color: #64748b;
    }

    /* =========================================================
       CARD ANALISIS
       ========================================================= */

    .analysis-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        min-height: 110px;
    }

    .analysis-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 6px;
    }

    .analysis-value {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
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
            background-color: #1e293b;
            border-color: #334155;
        }

        .spec-title {
            color: #94a3b8;
        }

        .spec-value {
            color: #f8fafc;
        }

        .spec-description {
            color: #cbd5e1;
        }

        .analysis-card {
            background-color: #1e293b;
            border-color: #334155;
        }

        .analysis-title {
            color: #94a3b8;
        }

        .analysis-value {
            color: #f8fafc;
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
    CR_val = engine_data["CR"]
except Exception as e:
    st.error(f"Gagal memuat dataset atau engine CBR: {e}")
    st.stop()


# ==============================================================================
# SIDEBAR — FORM INPUT USER
# ==============================================================================

st.sidebar.header("📝 Parameter Kebutuhan Ruangan")
st.sidebar.markdown(
    "Masukkan spesifikasi fisik ruangan, kondisi kebisingan, serta preferensi Anda."
)

with st.sidebar.form(key="user_input_form"):

    st.subheader("1. Karakteristik Fisik Ruangan")

    luas_ruangan_m2 = st.number_input(
        "Luas Lantai Ruangan (m²)",
        min_value=5.0,
        max_value=200.0,
        value=40.0,
        step=1.0,
        help="Luas area lantai ruangan sumber kebisingan.",
    )

    luas_dinding_m2 = st.number_input(
        "Luas Dinding yang Dipasangi Peredam (m²)",
        min_value=5.0,
        max_value=250.0,
        value=60.0,
        step=1.0,
        help="Luas bidang partisi/dinding yang akan dipasangi insulasi akustik.",
    )

    jenis_dinding = st.selectbox(
        "Jenis Struktur Dinding Saat Ini",
        options=["Hebel", "Batako", "Bata", "Beton", "Gypsum", "Kayu"],
        index=0,
    )

    ketebalan_dinding_cm = st.number_input(
        "Ketebalan Dinding Eksisting (cm)",
        min_value=5.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    st.subheader("2. Kondisi & Frekuensi Kebisingan")

    sumber_suara = st.selectbox(
        "Sumber Kebisingan Utama",
        options=[
            "Musik / audio",
            "Percakapan",
            "AC outdoor",
            "Home theater",
            "Studio band",
            "Lalu lintas",
            "Pompa air",
            "Kompresor",
            "Anak bermain",
            "Genset",
            "Bengkel las",
            "Karaoke",
            "Ruang rapat",
            "Mesin produksi",
        ],
        index=0,
    )

    kelas_frekuensi = st.selectbox(
        "Karakteristik Frekuensi Dominan",
        options=["low", "mid", "high"],
        index=1,
        format_func=lambda x: {
            "low": "Low / Rendah (Bass, Subwoofer, Drum, Genset < 250 Hz)",
            "mid": "Mid / Menengah (Vokal, Musik Akustik, Percakapan 250 Hz - 2 kHz)",
            "high": "High / Tinggi (Treble, Bising Mesin, Sirine > 2 kHz)",
        }[x],
        help="Karakteristik frekuensi menentukan jenis dan densitas material peredam yang efektif.",
    )

    st.subheader("3. Intensitas & Target Akustik")

    intensitas_suara_awal_db = st.slider(
        "Intensitas Suara Bising Awal (dB)",
        min_value=40.0,
        max_value=120.0,
        value=85.0,
        step=1.0,
        help="Tingkat kebisingan awal di dalam ruangan sumber suara.",
    )

    target_suara_db = st.slider(
        "Target Suara di Ruang Penerima (dB)",
        min_value=20.0,
        max_value=60.0,
        value=40.0,
        step=1.0,
        help="Standar kebisingan ruang hunian tenang berkisar 35 - 45 dB.",
    )

    st.subheader("4. Batasan Anggaran & Konfigurasi")

    kelas_anggaran = st.selectbox(
        "Kelas Anggaran Biaya",
        options=["Ekonomis", "Standar", "Premium"],
        index=1,
        help="Ekonomis: Rp 130k-180k/m² | Standar: Rp 170k-330k/m² | Premium: Rp 300k-700k/m²",
    )

    top_k_val = st.slider(
        "Jumlah Kasus Serupa (Top-K Retrieval)",
        min_value=3,
        max_value=15,
        value=10,
        step=1,
    )

    submit_button = st.form_submit_button(
        label="🔍 Analisis & Rekomendasikan Peredam",
        use_container_width=True,
    )


# ==============================================================================
# HALAMAN UTAMA
# ==============================================================================

st.title("🔊 Sistem Pakar Rekomendasi Desain Peredam Suara Ruangan")

st.markdown(
    """
    Sistem pakar ini mengintegrasikan **Improved Case-Based Reasoning (ICBR)** dengan 
    pembobotan hibrida **CRNHW (AHP + Shannon Entropy + SLSQP Optimization)**, 
    adaptasi kasus **CAHWM (SURA)** berbasis Grey Relational Analysis, serta pemodelan fisika 
    akustik lapangan baku **ASTM E336** & **Delany-Bazley Scaling**.
    """
)

st.divider()


# ==============================================================================
# PERSIAPAN INPUT DATA
# ==============================================================================

delta_db_calc = float(intensitas_suara_awal_db - target_suara_db)

user_input = {
    "luas_ruangan_m2": float(luas_ruangan_m2),
    "ketebalan_dinding_cm": float(ketebalan_dinding_cm),
    "luas_dinding_m2": float(luas_dinding_m2),
    "jenis_dinding": jenis_dinding,
    "sumber_suara": sumber_suara,
    "kelas_frekuensi": kelas_frekuensi,
    "kelas_anggaran": kelas_anggaran,
    "intensitas_suara_awal_db": float(intensitas_suara_awal_db),
    "target_suara_db": float(target_suara_db),
    "delta_db": delta_db_calc,
}


# ==============================================================================
# PROSES RETRIEVAL (PHASE 2)
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
# PROSES ADAPTASI SOLUSI (PHASE 3)
# ==============================================================================

try:
    adapted_result = cbr_engine.adapt_solution(
        top_k_candidates,
        user_input,
        Wh,
        df_cases_full=df_cases,
    )
except Exception as e:
    st.error(f"Gagal melakukan proses adaptasi solusi: {e}")
    st.stop()


# ==============================================================================
# SECTION 1 — RINGKASAN REKOMENDASI UTAMA
# ==============================================================================

st.subheader("🎯 Rekomendasi Solusi Peredam Suara Ruangan")

# Notifikasi Flanking Transmission Override Rule
if adapted_result["flanking_rule_applied"]:
    st.warning(
        "⚠️ **Aturan Batas Fisika (Flanking Transmission Override):** "
        f"Kebutuhan reduksi kebisingan sangat tinggi (ΔdB = {user_input['delta_db']:.1f} dB > 25 dB). "
        "Untuk mencegah kebocoran getaran mekanis melalui struktur dinding (*structural flanking*), "
        "sistem pemasangan otomatis disesuaikan menggunakan **Rongga Udara (Air Gap 5 cm) + Gypsum Decoupling 12 mm**."
    )

col1, col2 = st.columns(2)

with col1:
    # 1. Material & Tebal Snapped
    with st.container(border=True):
        st.markdown("### 🧱 Material Peredam Utama")
        st.markdown(f"## {adapted_result['material_peredam']}")
        st.markdown(
            f"Ketebalan Terpilih: **{adapted_result['ketebalan_peredam_mm']} mm** "
            f"*(Tersedia di pasar/dataset; hasil HWM kontinu: {adapted_result['tebal_hwm_kontinu_mm']} mm)*"
        )

    # 2. Sistem Pemasangan
    with st.container(border=True):
        st.markdown("### 🔧 Sistem Pemasangan")
        st.markdown(f"**{adapted_result['sistem_pemasangan']}**")
        st.caption(
            "Konfigurasi instalasi direkomendasikan guna meminimalkan jembatan getaran struktural."
        )

with col2:
    # 3. Estimasi Finansial
    with st.container(border=True):
        st.markdown("### 💰 Estimasi Total Biaya")
        st.markdown(f"## Rp {adapted_result['est_biaya_total_rp']:,}")
        st.markdown(
            f"Estimasi biaya satuan: **Rp {adapted_result['biaya_per_m2_rp']:,} / m²** "
            f"*(Kelas Anggaran: {user_input['kelas_anggaran']})*"
        )

    # 4. Evaluasi & Tingkat Rekomendasi
    with st.container(border=True):
        st.markdown("### 🎯 Evaluasi Kelayakan Rekomendasi")
        status_color = "🟢" if adapted_result["status_pemenuhan"] == "YA" else "🟡"
        st.markdown(
            f"## {status_color} {adapted_result['tingkat_rekomendasi']}"
        )
        st.markdown(
            f"• **Memenuhi Target Peredaman:** {adapted_result['status_pemenuhan']} "
            f"*(Rasio: {adapted_result['rasio_pemenuhan_pct']}%)*  \n"
            f"• **Kualitas Kemiripan Kasus Historis:** {adapted_result['max_similarity_pct']}%"
        )

st.divider()


# ==============================================================================
# SECTION 2 — DETAIL PERHITUNGAN AKUSTIK LAPANGAN (ASTM E336)
# ==============================================================================

st.subheader("📊 Detail Perhitungan Fisika Akustik Lapangan (ASTM E336 & Sabine)")

mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

with mcol1:
    with st.container(border=True):
        st.markdown("**Target Reduksi (Δ dB)**")
        st.markdown(f"### {user_input['delta_db']:.1f} dB")
        st.caption("Awal - Target")

with mcol2:
    with st.container(border=True):
        st.markdown("**NR Aktual Lapangan**")
        st.markdown(f"### {adapted_result['est_nr_aktual_db']:.1f} dB")
        st.caption(f"HWM tebal: {adapted_result['tl_material_hwm_db']} dB")

with mcol3:
    with st.container(border=True):
        st.markdown("**Koreksi Geometri (S/A)**")
        st.markdown(f"### {adapted_result['koreksi_geometri_db']:.2f} dB")
        st.caption("10·log₁₀(S / A_ruang)")

with mcol4:
    with st.container(border=True):
        st.markdown("**Est. Suara di Luar**")
        st.markdown(f"### {adapted_result['est_suara_luar_db']:.1f} dB")
        st.caption("Target ruang penerima")

with mcol5:
    with st.container(border=True):
        st.markdown("**Jarak Aman ke 40 dB**")
        st.markdown(f"### {adapted_result['jarak_hingga_40db_m']:.2f} m")
        st.caption("Inverse Square Law")

st.divider()


# ==============================================================================
# SECTION 3 — TOP-K KASUS BASIS DATA YANG PALING MIRIP
# ==============================================================================

st.subheader(
    f"📋 Top-{top_k_val} Kasus Basis Data yang Paling Mirip (Retrieval)"
)

display_df = top_k_candidates.copy()

display_df["similarity_score"] = (
    display_df["similarity_score"] * 100
).round(2).astype(str) + " %"

if "estimasi_biaya_rp" in display_df.columns:
    display_df["estimasi_biaya_rp"] = display_df["estimasi_biaya_rp"].apply(
        lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "-"
    )

cols_to_show = [
    "id",
    "jenis_dinding",
    "luas_dinding_m2",
    "sumber_suara",
    "kelas_frekuensi",
    "kelas_anggaran",
    "intensitas_suara_awal_db",
    "target_suara_db",
    "material_peredam",
    "ketebalan_peredam_mm",
    "estimasi_noise_reduction_db",
    "estimasi_biaya_rp",
    "similarity_score",
]
existing_cols = [c for c in cols_to_show if c in display_df.columns]
display_sub = display_df[existing_cols].copy()

display_sub = display_sub.rename(
    columns={
        "id": "ID",
        "jenis_dinding": "Dinding",
        "luas_dinding_m2": "Luas Dinding (m²)",
        "sumber_suara": "Sumber Suara",
        "kelas_frekuensi": "Frekuensi",
        "kelas_anggaran": "Anggaran",
        "intensitas_suara_awal_db": "Suara Awal (dB)",
        "target_suara_db": "Target (dB)",
        "material_peredam": "Material",
        "ketebalan_peredam_mm": "Tebal (mm)",
        "estimasi_noise_reduction_db": "NR Kasus (dB)",
        "estimasi_biaya_rp": "Biaya Historis (Rp)",
        "similarity_score": "Kemiripan",
    }
)

st.dataframe(
    display_sub,
    use_container_width=True,
    hide_index=True,
)


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption(
    "Sistem Pakar Peredam Suara Ruangan — Improved Case-Based Reasoning (ICBR) "
    f"| AHP Consistency Ratio (CR) = {CR_val:.4f} (< 0.10) | Standar ASTM E336"
)
