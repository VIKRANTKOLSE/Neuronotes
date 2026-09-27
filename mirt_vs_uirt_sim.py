"""
Monte Carlo Simulation: MIRT vs UIRT on a 10-item diagnostic test
==================================================================
Uses questions_final_qmatrix.csv item bank.
a_vector: 58-dim Q-matrix row (one value per concept node in the DAG).
We reduce to 3D by summing loadings into 3 ability dimensions
matching the CC-MIRT design (Dim0=foundational, Dim1=periodic/bonding, Dim2=coordination).
"""

import numpy as np
import pandas as pd
import ast
from sklearn.metrics import recall_score, accuracy_score

np.random.seed(42)

# ============================================================
# 1. DATA LOADING & Q-MATRIX PARSING
# ============================================================
df = pd.read_csv("../questions_final_qmatrix.csv")

def parse_vec(s):
    try:
        return np.array(ast.literal_eval(str(s)), dtype=float)
    except:
        return None

# Parse the 58-dim a_vector
df["a_full"] = df["a_vector"].apply(parse_vec)
df = df[df["a_full"].apply(lambda x: x is not None and len(x) == 58)].reset_index(drop=True)

# CC-MIRT dimension grouping (from cc_mirt.py):
# Dim 0: concept indices 0-5   (foundational)
# Dim 1: concept indices 6-33  (periodic/bonding)
# Dim 2: concept indices 34-57 (coordination/advanced)
DIM_SLICES = [
    slice(0, 6),    # Dim 0: foundational
    slice(6, 34),   # Dim 1: periodic/bonding
    slice(34, 58),  # Dim 2: coordination/advanced
]

def reduce_to_3d(a_full):
    """Sum loadings within each dimension group → 3D discrimination vector."""
    return np.array([a_full[s].sum() for s in DIM_SLICES])

df["a3d"] = df["a_full"].apply(reduce_to_3d)

# Filter: keep only items with at least one non-zero 3D loading
df = df[df["a3d"].apply(lambda v: np.any(v != 0))].reset_index(drop=True)
df["d_param"] = pd.to_numeric(df["d_param"], errors="coerce").fillna(0.0)

A_ALL = np.stack(df["a3d"].values)           # (N_items, 3)
D_ALL = df["d_param"].values.astype(float)   # (N_items,)
C_ALL = np.full(len(df), 0.25)               # fixed guessing

N_DIMS     = 3
N_ITEMS    = 10
N_STUDENTS = 1000

print(f"Item bank: {len(df)} usable items  |  {N_DIMS}D MIRT  |  {N_ITEMS}-item test  |  N={N_STUDENTS} students")

# ============================================================
# 2. SYNTHETIC COHORT
# ============================================================
mu    = np.zeros(N_DIMS)
rho   = 0.3
Sigma = rho * np.ones((N_DIMS, N_DIMS))
np.fill_diagonal(Sigma, 1.0)
THETA_TRUE = np.random.multivariate_normal(mu, Sigma, size=N_STUDENTS)  # (1000, 3)

# ============================================================
# 3. GROUND TRUTH DEFICIT LABELS
# ============================================================
DEFICIT_THRESHOLD = -0.5
Y_TRUE = (THETA_TRUE < DEFICIT_THRESHOLD).astype(int)  # (1000, 3)

# ============================================================
# 4. ITEM SELECTION (10 items spanning all 3 dims)
# ============================================================
selected_idx = []
# First: pick items with strongest loading per dimension
norms = np.linalg.norm(A_ALL, axis=1)
for dim in range(N_DIMS):
    ranked = np.argsort(A_ALL[:, dim])[::-1]
    for idx in ranked:
        if idx not in selected_idx:
            selected_idx.append(int(idx))
            break

# Fill rest by highest overall norm (diverse discrimination)
for idx in np.argsort(norms)[::-1]:
    if int(idx) not in selected_idx:
        selected_idx.append(int(idx))
    if len(selected_idx) == N_ITEMS:
        break

A_sel = A_ALL[selected_idx]   # (10, 3)
D_sel = D_ALL[selected_idx]   # (10,)
C_sel = C_ALL[selected_idx]   # (10,)

print(f"Selected items: {[df['source_id'].iloc[i] for i in selected_idx]}\n")

# ============================================================
# 5. RESPONSE SIMULATION (3PL CC-MIRT)
# ============================================================
def mirt_prob(theta, A, D, C):
    """theta: (N, 3), returns P: (N, n_items)"""
    logits = theta @ A.T + D
    pstar  = 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))
    return C + (1.0 - C) * pstar

P_TRUE = mirt_prob(THETA_TRUE, A_sel, D_sel, C_sel)  # (1000, 10)
R_OBS  = (np.random.rand(N_STUDENTS, N_ITEMS) < P_TRUE).astype(int)

# ============================================================
# 6. MAP ESTIMATION ENGINE
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def map_estimate(responses, A, D, C, n_dims, lr=0.08, n_iter=400, prior_var=1.0):
    """Gradient-descent MAP for theta given binary responses."""
    theta = np.zeros(n_dims)
    for _ in range(n_iter):
        if n_dims > 1:
            logit  = A @ theta + D
        else:
            logit  = A.ravel() * theta[0] + D
        pstar  = sigmoid(logit)
        p      = C + (1 - C) * pstar
        p      = np.clip(p, 1e-9, 1 - 1e-9)
        # Gradient of log-likelihood
        dp_dth = (1 - C) * pstar * (1 - pstar)   # (n_items,)
        resid  = responses - p                     # (n_items,)
        # grad wrt theta: sum_j [ resid_j / (p_j*(1-p_j)) * dp_dth_j * a_j ]
        w      = resid * dp_dth / np.clip(p * (1 - p), 1e-9, None)
        if n_dims > 1:
            grad_ll = A.T @ w                      # (n_dims,)
        else:
            grad_ll = np.array([np.sum(A.ravel() * w)])
        grad_prior = -theta / prior_var
        theta = np.clip(theta + lr * (grad_ll + grad_prior), -4, 4)
    return theta

# ---- MIRT Estimation (full 3D) ----
print("Estimating theta -- MIRT (3D)...")
THETA_MIRT = np.zeros((N_STUDENTS, N_DIMS))
for i in range(N_STUDENTS):
    THETA_MIRT[i] = map_estimate(R_OBS[i], A_sel, D_sel, C_sel, n_dims=N_DIMS)

# ---- UIRT Estimation (1D: scalar a = L2 norm of a_vec) ----
print("Estimating theta -- UIRT (1D)...")
A_uirt_scalar = np.linalg.norm(A_sel, axis=1)           # (10,)
THETA_UIRT_1D = np.zeros(N_STUDENTS)
for i in range(N_STUDENTS):
    THETA_UIRT_1D[i] = map_estimate(
        R_OBS[i], A_uirt_scalar, D_sel, C_sel, n_dims=1
    )[0]

# UIRT assigns the SAME 1D estimate to all 3 dimensions
THETA_UIRT = np.tile(THETA_UIRT_1D[:, None], (1, N_DIMS))   # (1000, 3)

# ============================================================
# 7. METRICS
# ============================================================
def rmse(true, est):
    return float(np.sqrt(np.mean((true - est) ** 2)))

def car_recall(theta_true, theta_est, thresh=DEFICIT_THRESHOLD):
    y_true = (theta_true < thresh).astype(int).flatten()
    y_pred = (theta_est  < thresh).astype(int).flatten()
    car = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return car, rec

rmse_uirt             = rmse(THETA_TRUE, THETA_UIRT)
rmse_mirt             = rmse(THETA_TRUE, THETA_MIRT)
car_uirt, rec_uirt    = car_recall(THETA_TRUE, THETA_UIRT)
car_mirt, rec_mirt    = car_recall(THETA_TRUE, THETA_MIRT)

# ============================================================
# 8. RESULTS TABLE
# ============================================================
sep = "=" * 62
print("\n" + sep)
print("   MONTE CARLO EVALUATION - MIRT vs UIRT (10-item, N=1,000)")
print(sep)
print(f"{'Metric':<35} {'UIRT (Baseline)':>13} {'MIRT (Proposed)':>13}")
print("-" * 62)
print(f"{'RMSE  (lower is better)':<35} {rmse_uirt:>13.4f} {rmse_mirt:>13.4f}")
print(f"{'CAR - Classification Accuracy':<35} {car_uirt:>12.2%} {car_mirt:>12.2%}")
print(f"{'Recall / Sensitivity':<35} {rec_uirt:>12.2%} {rec_mirt:>12.2%}")
print(sep)
print(f"\n  MIRT advantage over UIRT baseline:")
print(f"    RMSE  reduction : {rmse_uirt - rmse_mirt:+.4f}")
print(f"    CAR   gain      : {car_mirt  - car_uirt:+.2%}")
print(f"    Recall gain     : {rec_mirt  - rec_uirt:+.2%}")
print()
