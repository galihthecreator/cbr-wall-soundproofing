# ==============================================================================
# CBR ENGINE BACKEND (cbr_engine.py)
# Improved Case-Based Reasoning (ICBR) + CRNHW + CAHWM + ASTM E336
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ------------------------------------------------------------------------------
# KONSTANTA NAMA KOLOM & RENTANG PARAMETER
# ------------------------------------------------------------------------------
# delta_db dihapus dari FEATURE_COLS untuk mencegah multikolinearitas sempurna
# (Issue #7: rank matriks terjaga penuh, AHP 5x5 konsisten)
PROBLEM_NUM_COLS = [
    'luas_ruangan_m2',
    'ketebalan_dinding_cm',
    'luas_dinding_m2',
    'intensitas_suara_awal_db',
    'target_suara_db',
]

SOLUTION_NUM_COLS = [
    'ketebalan_peredam_mm',
    'estimasi_noise_reduction_db',
    'estimasi_suara_di_luar_db',
    'estimasi_biaya_rp',
]

# Rentang biaya acuan per kelas anggaran (Rp / m²)
BUDGET_RANGE = {
    'Ekonomis': (130000, 180000),
    'Standar': (170000, 330000),
    'Premium': (300000, 700000),
}


# ==============================================================================
# FUNGSI HELPER & FISIKA AKUSTIK
# ==============================================================================

def hitung_jarak_40db(suara_luar_db: float) -> float:
    """
    Menghitung jarak propagasi (meter) hingga intensitas suara meluruh ke ambang tenang 40 dB.
    Berdasarkan Inverse Square Law dalam medan bebas:
    SPL(r) = SPL(1m) - 20 * log10(r)  =>  r = 10 ** ((SPL - 40) / 20)
    Jika suara luar <= 40 dB, jarak aman adalah 0.0 meter (sudah memenuhi standar).
    """
    if suara_luar_db <= 40.0:
        return 0.0
    return round(float(10.0 ** ((suara_luar_db - 40.0) / 20.0)), 2)


def tandai_outlier_nr(df_in: pd.DataFrame, iqr_multiplier: float = 1.2) -> pd.DataFrame:
    """
    Mendeteksi kasus anomali (outlier) pada nilai Noise Reduction (NR) per kombinasi
    material dan ketebalan menggunakan metode Interquartile Range (IQR).
    Kasus outlier diberi tanda is_outlier_nr = True untuk diberikan penalti saat retrieval.
    """
    df_res = df_in.copy()
    df_res['is_outlier_nr'] = False
    for (mat, tebal), group in df_res.groupby(['material_peredam', 'ketebalan_peredam_mm']):
        if len(group) >= 3:
            q25 = group['estimasi_noise_reduction_db'].quantile(0.25)
            q75 = group['estimasi_noise_reduction_db'].quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - iqr_multiplier * iqr
                upper = q75 + iqr_multiplier * iqr
                idx = group[
                    (group['estimasi_noise_reduction_db'] < lower) |
                    (group['estimasi_noise_reduction_db'] > upper)
                ].index
                df_res.loc[idx, 'is_outlier_nr'] = True
    return df_res


def snap_tebal_ke_dataset(material: str, tebal_hwm: float, df_cases: pd.DataFrame) -> int:
    """
    Memilih ketebalan peredam nyata yang tersedia di dataset untuk material tertentu (Issue #3 & #11).
    Aturan: Memilih tebal tersedia >= tebal_hwm (ceiling match).
    Jika tebal_hwm melebihi tebal maksimum yang ada di dataset, gunakan tebal maksimum tersebut.
    Jika material tidak ditemukan di dataset, fallback pembulatan kelipatan 5 mm terdekat.
    """
    sub_df = df_cases[df_cases['material_peredam'].str.lower() == str(material).lower()]
    if len(sub_df) == 0:
        return int(max(5, round(tebal_hwm / 5.0) * 5))

    available = sorted(sub_df['ketebalan_peredam_mm'].unique())
    suitable = [t for t in available if t >= tebal_hwm]
    if suitable:
        return int(suitable[0])
    return int(available[-1])


def estimasi_biaya_dari_dataset(
    material: str, tebal_mm: int, luas_dinding_m2: float, df_cases: pd.DataFrame
) -> tuple[int, int]:
    """
    Menghitung estimasi total biaya (Rp) dan biaya per m² berdasarkan rata-rata harga per m²
    dari kasus historis pada dataset untuk material dan ketebalan terpilih (Issue #10).
    Dilakukan interpolasi linear jika ketebalan berada di antara nilai historis.
    Hasil total biaya dibulatkan ke kelipatan Rp 50.000 terdekat.
    """
    sub_df = df_cases[df_cases['material_peredam'].str.lower() == str(material).lower()].copy()
    if len(sub_df) == 0:
        sub_df = df_cases.copy()

    sub_df['harga_m2'] = sub_df['estimasi_biaya_rp'] / sub_df['luas_dinding_m2']
    summary = sub_df.groupby('ketebalan_peredam_mm')['harga_m2'].mean().sort_index()

    xp = summary.index.values.astype(float)
    fp = summary.values.astype(float)

    if len(xp) == 1:
        harga_m2_pred = fp[0]
    else:
        harga_m2_pred = float(np.interp(tebal_mm, xp, fp))

    total_biaya = int(np.round((harga_m2_pred * luas_dinding_m2) / 50000.0) * 50000)
    biaya_per_m2 = int(round(total_biaya / luas_dinding_m2))
    return total_biaya, biaya_per_m2


# ==============================================================================
# 1. LOAD & INITIALIZE ENGINE (PHASE 1 & PHASE 2)
# ==============================================================================
def load_and_initialize_engine(csv_path: str) -> dict:
    df_cases = pd.read_csv(csv_path)

    # Pastikan delta_db ada di dataset (dihitung untuk validasi target dan kendala)
    if 'delta_db' not in df_cases.columns:
        df_cases['delta_db'] = (
            df_cases['intensitas_suara_awal_db'] - df_cases['target_suara_db']
        )

    # Deteksi dan tandai outlier NR (Issue #14)
    df_cases = tandai_outlier_nr(df_cases)

    # Tangani missing values pada fitur numerik via median
    for col in PROBLEM_NUM_COLS:
        if col not in df_cases.columns:
            raise ValueError(f"Kolom numerik '{col}' tidak ditemukan di dataset!")
        if df_cases[col].isnull().any():
            df_cases[col] = df_cases[col].fillna(df_cases[col].median())

    # Ekstraksi matriks fitur numerik (5 fitur)
    X = df_cases[PROBLEM_NUM_COLS].values.astype(float)
    v_min = X.min(axis=0)
    v_max = X.max(axis=0)
    denom = np.where((v_max - v_min) == 0, 1.0, (v_max - v_min))
    X_norm = (X - v_min) / denom

    # --- A. AHP SUBJECTIVE WEIGHTS (Ws) -> Matriks 5x5 (Issue #7) ---
    # [0] luas_ruangan_m2, [1] ketebalan_dinding_cm, [2] luas_dinding_m2,
    # [3] intensitas_suara_awal_db, [4] target_suara_db
    AHP_MATRIX = np.array([
        [1.00, 2.00, 0.33, 0.25, 0.33],  # [0] luas_ruangan_m2
        [0.50, 1.00, 0.25, 0.20, 0.25],  # [1] ketebalan_dinding_cm
        [3.00, 4.00, 1.00, 0.50, 1.00],  # [2] luas_dinding_m2
        [4.00, 5.00, 2.00, 1.00, 2.00],  # [3] intensitas_suara_awal_db
        [3.00, 4.00, 1.00, 0.50, 1.00],  # [4] target_suara_db
    ])

    m = AHP_MATRIX.shape[0]
    geom_mean = np.exp(np.mean(np.log(AHP_MATRIX), axis=1))
    Ws = geom_mean / np.sum(geom_mean)

    lambda_max = np.mean((AHP_MATRIX @ Ws) / Ws)
    CI = (lambda_max - m) / (m - 1) if m > 1 else 0.0
    RI_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.89, 5: 1.12, 6: 1.26}
    CR = CI / RI_table.get(m, 1.12)

    # --- B. SHANNON ENTROPY OBJECTIVE WEIGHTS (Wo) ---
    eps = 1e-12
    col_sums = np.sum(X_norm, axis=0) + eps
    Q = X_norm / col_sums
    e_o = -1.0 / np.log(len(df_cases)) * np.sum(Q * np.log(Q + eps), axis=0)
    g_j = 1.0 - e_o
    Wo = g_j / np.sum(g_j)

    # --- C. OPTIMASI HIBRIDA SLSQP (Wh) ---
    def objective_func(Wh_var):
        dev_s = np.sum(((Wh_var - Ws) * X_norm) ** 2)
        dev_o = np.sum(((Wh_var - Wo) * X_norm) ** 2)
        return dev_s + dev_o

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(m)]
    opt_res = minimize(
        objective_func,
        (Ws + Wo) / 2.0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )
    Wh = opt_res.x if opt_res.success else (0.5 * Ws + 0.5 * Wo)
    Wh = Wh / np.sum(Wh)

    return {
        'df_cases': df_cases,
        'Wh': Wh,
        'Ws': Ws,
        'Wo': Wo,
        'CR': float(CR),
        'v_min': v_min,
        'v_max': v_max,
        'v_denom': denom,
    }


# ==============================================================================
# 2. RETRIEVE TOP-K CANDIDATES (PHASE 2)
# ==============================================================================
def retrieve_top_k(
    df_cases: pd.DataFrame,
    user_input: dict,
    Wh: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    K: int = 10,
) -> pd.DataFrame:
    """
    Mengambil K kasus paling mirip menggunakan 5 komponen similarity (Issue #4 & #8):
    total_sim = 0.60 * sim_num + 0.10 * sim_dinding + 0.10 * sim_sumber + 0.10 * sim_frek + 0.10 * sim_anggaran
    Dan menerapkan penalti 0.70 untuk kasus outlier (Issue #14).
    """
    if 'delta_db' not in user_input or user_input['delta_db'] is None:
        user_input['delta_db'] = (
            user_input['intensitas_suara_awal_db'] - user_input['target_suara_db']
        )

    # 1. Similarity Numerik (5 fitur)
    user_vector = np.array([user_input[col] for col in PROBLEM_NUM_COLS], dtype=float)
    denom = np.where((v_max - v_min) == 0, 1.0, (v_max - v_min))
    u_norm = np.clip((user_vector - v_min) / denom, 0.0, 1.0)

    X_raw = df_cases[PROBLEM_NUM_COLS].values.astype(float)
    X_norm = np.clip((X_raw - v_min) / denom, 0.0, 1.0)

    diff_matrix = np.abs(X_norm - u_norm)
    sim_num_matrix = 1.0 - diff_matrix
    weighted_sim_num = np.dot(sim_num_matrix, Wh) / np.sum(Wh)

    # 2. Similarity Kategorikal
    user_dinding = str(user_input.get('jenis_dinding', 'hebel')).lower()
    user_sumber = str(user_input.get('sumber_suara', 'musik / audio')).lower()
    user_frek = str(user_input.get('kelas_frekuensi', 'mid')).lower()
    user_anggaran = str(user_input.get('kelas_anggaran', 'Standar')).lower()

    sim_cat_dinding = (df_cases['jenis_dinding'].astype(str).str.lower().values == user_dinding).astype(float)
    sim_cat_sumber = (df_cases['sumber_suara'].astype(str).str.lower().values == user_sumber).astype(float)
    sim_cat_frek = (df_cases['kelas_frekuensi'].astype(str).str.lower().values == user_frek).astype(float)
    sim_cat_anggaran = (df_cases['kelas_anggaran'].astype(str).str.lower().values == user_anggaran).astype(float)

    # 3. Formulasi Total Similarity Terintegrasi (0.60 / 0.10 / 0.10 / 0.10 / 0.10)
    total_similarity = (
        0.60 * weighted_sim_num +
        0.10 * sim_cat_dinding +
        0.10 * sim_cat_sumber +
        0.10 * sim_cat_frek +
        0.10 * sim_cat_anggaran
    )

    # 4. Outlier Penalty (Issue #14)
    if 'is_outlier_nr' in df_cases.columns:
        total_similarity = np.where(df_cases['is_outlier_nr'], total_similarity * 0.70, total_similarity)

    df_result = df_cases.copy()
    df_result['similarity_score'] = total_similarity
    df_result['similarity_pct'] = np.round(total_similarity * 100, 2)

    top_k_candidates = df_result.sort_values(
        by='similarity_score', ascending=False
    ).head(K)

    return top_k_candidates.reset_index(drop=True)


# ==============================================================================
# 3. CASE ADAPTATION ENGINE (PHASE 3: CAHWM + ASTM E336 REVISED)
# ==============================================================================
def adapt_solution(
    top_candidates: pd.DataFrame,
    user_input: dict,
    Wh: np.ndarray,
    df_cases_full: pd.DataFrame = None,
) -> dict:
    """
    Melakukan proses adaptasi kasus menggunakan pendekatan CAHWM (SURA) dengan:
    - Matriks A terdeferensiasi per atribut (tebal, NR, suara luar, anggaran) (Issue #5 & #8)
    - Eliminasi double-counting koreksi geometri (Issue #1)
    - Snapping ketebalan peredam ke nilai valid dataset (Issue #3 & #11)
    - Rekalkulasi NR lapangan via Delany-Bazley sqrt scaling (Issue #9)
    - Rekalkulasi biaya berbasis dataset historis (Issue #10)
    - Penentuan Tingkat Rekomendasi berdasarkan rasio pemenuhan target NR (Issue #2)
    """
    eps = 1e-12
    K_candidates = len(top_candidates)
    m_prob = len(PROBLEM_NUM_COLS)
    n_sol = len(SOLUTION_NUM_COLS)

    if df_cases_full is None:
        df_cases_full = top_candidates

    # Pastikan delta_db tersedia
    if 'delta_db' not in user_input or user_input['delta_db'] is None:
        user_input['delta_db'] = (
            user_input['intensitas_suara_awal_db'] - user_input['target_suara_db']
        )

    # --- STEP 3: MATRIKS KEMIRIPAN (S) & VEKTOR UTILITAS (U) ---
    S_raw = top_candidates[PROBLEM_NUM_COLS].values.astype(float)
    s_min = S_raw.min(axis=0)
    s_max = S_raw.max(axis=0)
    s_denom = np.where((s_max - s_min) == 0, 1.0, (s_max - s_min))
    S = (S_raw - s_min) / s_denom

    u_k = np.dot(S, Wh)
    u_norm = u_k / (np.sum(u_k) + eps)

    # --- STEP 4: GREY RELATIONAL ANALYSIS MATRIX (R) ---
    p_2 = S_raw / (S_raw[0, :] + eps)
    Sol_raw = top_candidates[SOLUTION_NUM_COLS].values.astype(float)
    s_2 = Sol_raw / (Sol_raw[0, :] + eps)

    xi = 0.5
    R = np.zeros((m_prob, n_sol))
    for i in range(m_prob):
        for j in range(n_sol):
            diff = np.abs(p_2[:, i] - s_2[:, j])
            min_diff = np.min(diff)
            max_diff = np.max(diff)
            grey_coef = (min_diff + xi * max_diff) / (diff + xi * max_diff + eps)
            R[i, j] = np.sum(u_norm * grey_coef)

    # --- STEP 5: MATRIKS ADAPTABILITAS (A) PER ATRIBUT (Issue #5 & #8) ---
    sim_scores = top_candidates['similarity_score'].values
    A = np.zeros((K_candidates, n_sol))

    user_budget = user_input.get('kelas_anggaran', 'Standar')
    max_budget_m2 = BUDGET_RANGE.get(user_budget, (0, 330000))[1]

    for k_idx in range(K_candidates):
        row_cand = top_candidates.iloc[k_idx]

        # 1. Penalti Ketebalan (kolom 0): jika tebal > 75mm padahal luas ruangan kecil (< 25 m²)
        penalty_tebal = 0.85 if (
            row_cand['ketebalan_peredam_mm'] > 75 and user_input['luas_ruangan_m2'] < 25.0
        ) else 1.0

        # 2. Penalti NR (kolom 1): jika NR kandidat di bawah target peredaman ΔdB
        penalty_nr = 0.90 if row_cand['estimasi_noise_reduction_db'] < user_input['delta_db'] else 1.0

        # 3. Penalti Suara Luar (kolom 2): jika suara di luar melebihi target kenyamanan user
        penalty_suara = 0.90 if row_cand['estimasi_suara_di_luar_db'] > user_input['target_suara_db'] else 1.0

        # 4. Penalti Biaya (kolom 3): penalti proporsional jika harga/m² melebihi batas kelas anggaran
        cand_biaya_m2 = row_cand['estimasi_biaya_rp'] / row_cand['luas_dinding_m2']
        if cand_biaya_m2 > max_budget_m2:
            over_ratio = (cand_biaya_m2 - max_budget_m2) / max_budget_m2
            penalty_biaya = max(0.60, 1.0 - over_ratio)
        else:
            penalty_biaya = 1.0

        penalties = [penalty_tebal, penalty_nr, penalty_suara, penalty_biaya]
        for j in range(n_sol):
            A[k_idx, j] = 0.5 * sim_scores[k_idx] + 0.5 * penalties[j]

    # --- STEP 6: HYBRID WEIGHTED MEAN (HWM) ---
    SR_matrix = np.dot(S, R)
    WM = SR_matrix * A
    HWM = WM / (np.sum(WM, axis=0) + eps)

    adapted_numerical = np.sum(HWM * Sol_raw, axis=0)
    adapted_sol_dict = {
        col: adapted_numerical[idx] for idx, col in enumerate(SOLUTION_NUM_COLS)
    }

    # Adaptasi atribut kategorikal melalui weighted voting
    def adapt_categorical(top_df, col_name):
        weighted_votes = {}
        for _, row in top_df.iterrows():
            val = row[col_name]
            score = row['similarity_score']
            weighted_votes[val] = weighted_votes.get(val, 0.0) + score
        return max(weighted_votes, key=weighted_votes.get)

    adapted_material = adapt_categorical(top_candidates, 'material_peredam')
    adapted_sistem = adapt_categorical(top_candidates, 'sistem_pemasangan')

    # Flanking transmission override rule (ASTM E336 boundary)
    flanking_applied = False
    if user_input['delta_db'] > 25.0:
        if adapted_sistem == 'Ditempel langsung ke dinding':
            adapted_sistem = 'Peredam + rongga udara (air gap 5 cm) + gypsum 12mm'
            flanking_applied = True

    # Informasi koreksi geometri ruangan (ASTM E336)
    S_user = user_input['luas_dinding_m2']
    A_user = user_input['luas_ruangan_m2'] * 0.20 + 5.0
    koreksi_geometri_user = round(10.0 * np.log10(S_user / A_user), 2)

    # --- PERBAIKAN ASTM E336, SNAPPING, & REKALKULASI (Issue #1, #3, #9, #10) ---
    tebal_hwm = float(adapted_sol_dict['ketebalan_peredam_mm'])
    nr_lapangan_hwm = float(adapted_sol_dict['estimasi_noise_reduction_db'])

    # 1. Snap tebal ke dataset historis (Issue #3 & #11)
    tebal_mm = snap_tebal_ke_dataset(adapted_material, tebal_hwm, df_cases_full)

    # 2. Rekalkulasi NR via Delany-Bazley sqrt scaling (Issue #9)
    # Catatan: Tidak mengurangi koreksi geometri lagi karena nilai di dataset sudah NR lapangan (Issue #1)
    nr_lapangan_final = round(nr_lapangan_hwm * ((tebal_mm / (tebal_hwm + 1e-12)) ** 0.5), 1)

    # 3. Estimasi Suara Luar & Jarak Aman hingga 40 dB (Issue #12)
    est_suara_luar = max(
        0.0, round(user_input['intensitas_suara_awal_db'] - nr_lapangan_final, 1)
    )
    jarak_40db = hitung_jarak_40db(est_suara_luar)

    # 4. Rekalkulasi Total Biaya & Biaya per m² dari dataset (Issue #10)
    est_biaya_total, biaya_per_m2 = estimasi_biaya_dari_dataset(
        adapted_material, tebal_mm, user_input['luas_dinding_m2'], df_cases_full
    )

    # 5. Tingkat Rekomendasi berbasis rasio pemenuhan target reduksi (Issue #2)
    rasio_pemenuhan = nr_lapangan_final / (user_input['delta_db'] + 1e-12)
    status_pemenuhan = "YA" if nr_lapangan_final >= user_input['delta_db'] else "TIDAK"

    if rasio_pemenuhan >= 1.05:
        tingkat_rekomendasi = 'Tinggi'
    elif rasio_pemenuhan >= 0.90:
        tingkat_rekomendasi = 'Sedang'
    else:
        tingkat_rekomendasi = 'Rendah'

    max_sim = float(top_candidates['similarity_score'].max())

    return {
        'material_peredam': adapted_material,
        'sistem_pemasangan': adapted_sistem,
        'ketebalan_peredam_mm': tebal_mm,
        'tebal_hwm_kontinu_mm': round(tebal_hwm, 2),
        'tl_material_hwm_db': round(nr_lapangan_hwm, 1),
        'koreksi_geometri_db': koreksi_geometri_user,
        'est_nr_aktual_db': nr_lapangan_final,
        'est_suara_luar_db': est_suara_luar,
        'jarak_hingga_40db_m': jarak_40db,
        'est_biaya_total_rp': est_biaya_total,
        'biaya_per_m2_rp': biaya_per_m2,
        'tingkat_rekomendasi': tingkat_rekomendasi,
        'status_pemenuhan': status_pemenuhan,
        'rasio_pemenuhan_pct': round(rasio_pemenuhan * 100, 2),
        'max_similarity_pct': round(max_sim * 100, 2),
        'flanking_rule_applied': flanking_applied,
        'matriks_A': A,
    }
