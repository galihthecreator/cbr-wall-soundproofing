# ==============================================================================
# CBR ENGINE BACKEND (cbr_engine.py)
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ------------------------------------------------------------------------------
# KONSTANTA NAMA KOLOM (PASTIKAN PROBLEM_NUM_COLS ADA 6 ELEMEN)
# ------------------------------------------------------------------------------
PROBLEM_NUM_COLS = [
    'luas_ruangan_m2',
    'ketebalan_dinding_cm',
    'luas_dinding_m2',
    'intensitas_suara_awal_db',
    'target_suara_db',
    'delta_db',  # Kolom ke-6
]

SOLUTION_NUM_COLS = [
    'ketebalan_peredam_mm',
    'estimasi_noise_reduction_db',
    'estimasi_suara_di_luar_db',
    'estimasi_biaya_rp',
]


# ==============================================================================
# 1. LOAD & INITIALIZE ENGINE (PHASE 1)
# ==============================================================================
def load_and_initialize_engine(csv_path: str):
  df_cases = pd.read_csv(csv_path)

  # Hitung delta_db jika belum ada di dataset CSV
  if 'delta_db' not in df_cases.columns:
    df_cases['delta_db'] = (
        df_cases['intensitas_suara_awal_db'] - df_cases['target_suara_db']
    )

  # Validasi kelengkapan kolom
  for col in PROBLEM_NUM_COLS + SOLUTION_NUM_COLS:
    if col not in df_cases.columns:
      raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset!")

  # Extract Matriks X (Shape: N x 6)
  X = df_cases[PROBLEM_NUM_COLS].values.astype(float)
  v_min = X.min(axis=0)  # Shape (6,)
  v_max = X.max(axis=0)  # Shape (6,)

  # --- A. AHP WEIGHTS (Ws) -> Shape (6,) ---
  AHP_MATRIX = np.array([
      [1.0, 3.0, 2.0, 1 / 3, 1 / 3, 1 / 4],
      [1 / 3, 1.0, 1 / 2, 1 / 4, 1 / 4, 1 / 5],
      [1 / 2, 2.0, 1.0, 1 / 3, 1 / 3, 1 / 4],
      [3.0, 4.0, 3.0, 1.0, 1.0, 1 / 2],
      [3.0, 4.0, 3.0, 1.0, 1.0, 1 / 2],
      [4.0, 5.0, 4.0, 2.0, 2.0, 1.0],
  ])

  eigvals, eigvecs = np.linalg.eig(AHP_MATRIX)
  max_index = np.argmax(np.real(eigvals))
  Ws = np.real(eigvecs[:, max_index])
  Ws = Ws / np.sum(Ws)  # Shape (6,)

  # --- B. SHANNON ENTROPY WEIGHTS (Wo) -> Shape (6,) ---
  eps = 1e-12
  denom = np.where((v_max - v_min) == 0, 1.0, (v_max - v_min))
  X_norm = (X - v_min) / denom

  P = X_norm / (np.sum(X_norm, axis=0) + eps)
  E = -1.0 / np.log(len(df_cases)) * np.sum(P * np.log(P + eps), axis=0)
  d = 1.0 - E
  Wo = d / np.sum(d)  # Shape (6,)

  # --- C. OPTIMASI HIBRIDA SLSQP (Wh) ---
  def objective_func(alpha):
    w_h = alpha[0] * Ws + (1.0 - alpha[0]) * Wo
    return np.var(w_h)

  res = minimize(
      objective_func,
      x0=[0.5],
      bounds=[(0.0, 1.0)],
      method='SLSQP',
  )
  alpha_opt = float(res.x[0]) if res.success else 0.5
  Wh = alpha_opt * Ws + (1.0 - alpha_opt) * Wo
  Wh = Wh / np.sum(Wh)  # Shape (6,)

  return {
      'df_cases': df_cases,
      'Wh': Wh,
      'Ws': Ws,
      'Wo': Wo,
      'alpha_opt': alpha_opt,
      'v_min': v_min,
      'v_max': v_max,
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
):
  if 'delta_db' not in user_input or user_input['delta_db'] is None:
    user_input['delta_db'] = (
        user_input['intensitas_suara_awal_db'] - user_input['target_suara_db']
    )

  user_vector = np.array([user_input[col] for col in PROBLEM_NUM_COLS], dtype=float)

  denom = np.where((v_max - v_min) == 0, 1.0, (v_max - v_min))
  u_norm = np.clip((user_vector - v_min) / denom, 0.0, 1.0)

  X_raw = df_cases[PROBLEM_NUM_COLS].values.astype(float)
  X_norm = np.clip((X_raw - v_min) / denom, 0.0, 1.0)

  diff_sq = (X_norm - u_norm) ** 2
  weighted_dist = np.sqrt(np.dot(diff_sq, Wh))
  similarity_scores = 1.0 - weighted_dist

  df_result = df_cases.copy()
  df_result['similarity_score'] = similarity_scores

  top_k_candidates = df_result.sort_values(
      by='similarity_score', ascending=False
  ).head(K)

  return top_k_candidates.reset_index(drop=True)


# ==============================================================================
# 3. CASE ADAPTATION ENGINE (PHASE 3)
# ==============================================================================
def adapt_solution(
    top_candidates: pd.DataFrame, user_input: dict, Wh: np.ndarray
):
  eps = 1e-12
  K_candidates = len(top_candidates)
  m_prob = len(PROBLEM_NUM_COLS)
  n_sol = len(SOLUTION_NUM_COLS)

  S_raw = top_candidates[PROBLEM_NUM_COLS].values.astype(float)
  s_min = S_raw.min(axis=0)
  s_max = S_raw.max(axis=0)
  s_denom = np.where((s_max - s_min) == 0, 1.0, (s_max - s_min))
  S = (S_raw - s_min) / s_denom

  u_k = np.dot(S, Wh)
  u_norm = u_k / (np.sum(u_k) + eps)

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

  sim_scores = top_candidates['similarity_score'].values
  A = np.zeros((K_candidates, n_sol))

  for k_idx in range(K_candidates):
    row_cand = top_candidates.iloc[k_idx]
    constraint_penalty = 1.0

    if (
        row_cand['ketebalan_peredam_mm'] > 75
        and user_input['luas_ruangan_m2'] < 25.0
    ):
      constraint_penalty *= 0.85

    if row_cand['estimasi_noise_reduction_db'] < user_input['delta_db']:
      constraint_penalty *= 0.90

    for j in range(n_sol):
      A[k_idx, j] = 0.5 * sim_scores[k_idx] + 0.5 * constraint_penalty

  SR_matrix = np.dot(S, R)
  WM = SR_matrix * A
  HWM = WM / (np.sum(WM, axis=0) + eps)

  adapted_numerical = np.sum(HWM * Sol_raw, axis=0)
  adapted_sol_dict = {
      col: adapted_numerical[idx] for idx, col in enumerate(SOLUTION_NUM_COLS)
  }

  def adapt_categorical(top_df, col_name):
    weighted_votes = {}
    for _, row in top_df.iterrows():
      val = row[col_name]
      score = row['similarity_score']
      weighted_votes[val] = weighted_votes.get(val, 0.0) + score
    return max(weighted_votes, key=weighted_votes.get)

  adapted_material = adapt_categorical(top_candidates, 'material_peredam')
  adapted_sistem = adapt_categorical(top_candidates, 'sistem_pemasangan')

  flanking_applied = False
  if user_input['delta_db'] > 25.0:
    if adapted_sistem == 'Ditempel langsung ke dinding':
      adapted_sistem = (
          'Peredam + rongga udara (air gap 5 cm) + gypsum 12mm'
      )
      flanking_applied = True

  S_user = user_input['luas_dinding_m2']
  A_user = user_input['luas_ruangan_m2'] * 0.20 + 5.0
  koreksi_geometri = 10 * np.log10(S_user / A_user)

  tl_material_hwm = round(
      adapted_sol_dict['estimasi_noise_reduction_db'], 1
  )
  est_nr_aktual = round(tl_material_hwm - koreksi_geometri, 1)
  est_suara_luar = max(
      0.0, round(user_input['intensitas_suara_awal_db'] - est_nr_aktual, 1)
  )

  tebal_mm = int(round(adapted_sol_dict['ketebalan_peredam_mm'] / 5) * 5)
  est_biaya_total = int(
      np.round(adapted_sol_dict['estimasi_biaya_rp'] / 50000) * 50000
  )
  biaya_per_m2 = int(round(est_biaya_total / user_input['luas_dinding_m2']))

  max_sim = float(top_candidates['similarity_score'].max())
  if max_sim >= 0.85:
    tingkat_rekomendasi = 'Tinggi'
  elif max_sim >= 0.70:
    tingkat_rekomendasi = 'Sedang'
  else:
    tingkat_rekomendasi = 'Rendah'

  return {
      'material_peredam': adapted_material,
      'sistem_pemasangan': adapted_sistem,
      'ketebalan_peredam_mm': tebal_mm,
      'tl_material_hwm_db': tl_material_hwm,
      'koreksi_geometri_db': round(koreksi_geometri, 2),
      'est_nr_aktual_db': est_nr_aktual,
      'est_suara_luar_db': est_suara_luar,
      'est_biaya_total_rp': est_biaya_total,
      'biaya_per_m2_rp': biaya_per_m2,
      'tingkat_rekomendasi': tingkat_rekomendasi,
      'max_similarity_pct': round(max_sim * 100, 2),
      'flanking_rule_applied': flanking_applied,
  }