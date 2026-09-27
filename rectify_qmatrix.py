"""
rectify_qmatrix.py - Q-Matrix Dataset Rectification Pipeline
==============================================================
Applies 4 mathematical/structural fixes to questions_final_qmatrix.csv
and outputs questions_calibrated_qmatrix.csv.

Fixes:
  1. Zero-vector enforcement for correct option z-vectors
  2. Unicode/symbol cleanup (? -> sigma, delta, etc.)
  3. Procedural slip noise injection (~15% of wrong options)
  4. 58-concept diagnostic report

Run:  py rectify_qmatrix.py
"""

import ast
import io
import json
import math
import random
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- Configuration ---------------------------------------------------------
SEED = 42
SLIP_INJECTION_RATE = 0.15   # 15% of wrong options get their z-vector zeroed
INPUT_CSV  = Path("data/questions_final_qmatrix.csv")
OUTPUT_CSV = Path("data/questions_calibrated_qmatrix.csv")

random.seed(SEED)
np.random.seed(SEED)

# -- Load ------------------------------------------------------------------
print("=" * 70)
print("  MIRT Q-Matrix Rectification Pipeline")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)
N_ITEMS = len(df)
print(f"\n[LOAD] Read {N_ITEMS} items from {INPUT_CSV}")
print(f"       Columns: {list(df.columns)}")

# -- Helper: parse z-vector string -> list ----------------------------------
def parse_z(z_str):
    """Parse a z-vector string like '[0, 1, 0, ...]' into a Python list of ints."""
    try:
        return list(ast.literal_eval(z_str))
    except (ValueError, SyntaxError):
        return [0] * 15

def z_to_str(z_list):
    """Convert a list of ints back to the CSV string format."""
    return str(z_list)


# ===========================================================================
# FIX 1: Zero-vector enforcement for correct option z-vectors
# ===========================================================================
print("\n" + "-" * 70)
print("FIX 1: Zeroing z-vectors for correct options")
print("-" * 70)

fix1_count = 0
fix1_details = []

for idx, row in df.iterrows():
    correct_opt = int(row['correct_option'])
    z_col = f'z{correct_opt}'
    z_vec = parse_z(row[z_col])
    
    if any(v != 0 for v in z_vec):
        fix1_count += 1
        if fix1_count <= 5:  # Log first 5 examples
            fix1_details.append(f"  {row['source_id']}: z{correct_opt} = {z_vec} -> [0]*15")
        # Zero out the correct option's z-vector
        df.at[idx, z_col] = z_to_str([0] * 15)

print(f"  Fixed: {fix1_count}/{N_ITEMS} items had non-zero correct-option z-vectors")
for d in fix1_details:
    print(d)
if fix1_count > 5:
    print(f"  ... and {fix1_count - 5} more")


# ===========================================================================
# FIX 2: Unicode/Symbol Cleanup
# ===========================================================================
print("\n" + "-" * 70)
print("FIX 2: Cleaning corrupted math symbols")
print("-" * 70)

# Replacement rules: context-specific substitutions
SYMBOL_REPLACEMENTS = [
    # Shielding constant sigma: (?) -> (σ)
    (r'shielding constant \(\?\)', 'shielding constant (σ)'),
    (r'shielding constant for (.+?) \?', r'shielding constant for \1 σ'),
    # Z - ? -> Z - σ (Slater's rule context)
    (r'Z - \?', 'Z - σ'),
    (r'Z minus \?', 'Z minus σ'),
    # Crystal field splitting (?) -> (Δ)
    (r'crystal field splitting \(\?\)', 'crystal field splitting (Δ)'),
    (r'splitting parameter \(\?\)', 'splitting parameter (Δ)'),
    (r'splitting energy \(\?\)', 'splitting energy (Δ)'),
    # Stability constant patterns: K? -> Kₙ (stepwise constants)
    # ΔG° pattern
    (r'\?G\?', 'ΔG°'),
    (r'\?G°', 'ΔG°'),
    (r'\?H\?', 'ΔH°'),
    (r'\?S\?', 'ΔS°'),
    # σ in context of sigma notation
    (r'\?\? = K\?', 'βₙ = K₁'),
    # Approximation symbol: ? -> approx
    (r' \? (\d)', ' ' + chr(0x2248) + r' \1'),
]

import re

text_cols = ['question_text', 'opt1', 'opt2', 'opt3', 'opt4',
             'reason1', 'reason2', 'reason3', 'reason4']

fix2_count = 0
for col in text_cols:
    for pattern, replacement in SYMBOL_REPLACEMENTS:
        mask = df[col].str.contains(pattern, regex=True, na=False)
        n_matches = mask.sum()
        if n_matches > 0:
            df[col] = df[col].str.replace(pattern, replacement, regex=True)
            fix2_count += n_matches
            print(f"  {col}: {n_matches} replacements for pattern '{pattern}' -> '{replacement}'")

print(f"  Total text symbol fixes: {fix2_count}")


# ===========================================================================
# FIX 3: Procedural Slip Noise Injection
# ===========================================================================
print("\n" + "-" * 70)
print(f"FIX 3: Injecting procedural slip noise (~{SLIP_INJECTION_RATE*100:.0f}% of wrong options)")
print("-" * 70)

# Collect all wrong-option indices
wrong_option_indices = []  # list of (row_idx, z_col, original_z)
for idx, row in df.iterrows():
    correct_opt = int(row['correct_option'])
    for opt in range(1, 5):
        if opt == correct_opt:
            continue
        z_col = f'z{opt}'
        z_vec = parse_z(row[z_col])
        if any(v != 0 for v in z_vec):
            wrong_option_indices.append((idx, z_col, z_vec))

print(f"  Total wrong options with non-zero z-vectors: {len(wrong_option_indices)}")

# Randomly select ~15% to zero out (simulating procedural slips)
n_to_slip = int(round(SLIP_INJECTION_RATE * len(wrong_option_indices)))
slip_indices = random.sample(wrong_option_indices, min(n_to_slip, len(wrong_option_indices)))

fix3_count = 0
for (row_idx, z_col, orig_z) in slip_indices:
    df.at[row_idx, z_col] = z_to_str([0] * 15)
    fix3_count += 1

print(f"  Injected slip noise: {fix3_count}/{len(wrong_option_indices)} wrong options zeroed")
print(f"  Effective slip rate: {fix3_count/len(wrong_option_indices)*100:.1f}%")

# Verify: no item has ALL wrong options zeroed
all_zero_items = 0
for idx, row in df.iterrows():
    correct_opt = int(row['correct_option'])
    all_wrong_zero = True
    for opt in range(1, 5):
        if opt == correct_opt:
            continue
        z_vec = parse_z(row[f'z{opt}'])
        if any(v != 0 for v in z_vec):
            all_wrong_zero = False
            break
    if all_wrong_zero:
        all_zero_items += 1

if all_zero_items > 0:
    print(f"  !! WARNING: {all_zero_items} items have ALL wrong options zeroed — re-injecting one trap each")
    # Re-inject at least one non-zero z for these items from original data
    df_orig = pd.read_csv(INPUT_CSV)
    for idx, row in df.iterrows():
        correct_opt = int(row['correct_option'])
        all_wrong_zero = True
        for opt in range(1, 5):
            if opt == correct_opt:
                continue
            z_vec = parse_z(row[f'z{opt}'])
            if any(v != 0 for v in z_vec):
                all_wrong_zero = False
                break
        if all_wrong_zero:
            # Restore one random wrong option from original
            wrong_opts = [o for o in range(1, 5) if o != correct_opt]
            restore_opt = random.choice(wrong_opts)
            orig_z = parse_z(df_orig.at[idx, f'z{restore_opt}'])
            df.at[idx, f'z{restore_opt}'] = z_to_str(orig_z)
    # Re-check
    all_zero_items_after = 0
    for idx, row in df.iterrows():
        correct_opt = int(row['correct_option'])
        all_wrong_zero = True
        for opt in range(1, 5):
            if opt == correct_opt:
                continue
            z_vec = parse_z(row[f'z{opt}'])
            if any(v != 0 for v in z_vec):
                all_wrong_zero = False
                break
        if all_wrong_zero:
            all_zero_items_after += 1
    print(f"  After restoration: {all_zero_items_after} items with all-zero wrong options")
else:
    print(f"  OK Safety check passed: 0 items have all wrong options zeroed")


# ===========================================================================
# FIX 4: 58-Concept Diagnostic Report
# ===========================================================================
print("\n" + "-" * 70)
print("FIX 4: 58-Concept Diagnostic Report")
print("-" * 70)

# 4a. Concept coverage
concept_counts = df['concept'].value_counts()
print(f"\n  Unique concepts in dataset: {df['concept'].nunique()}")
print(f"  Items per concept (min/max/mean): {concept_counts.min()}/{concept_counts.max()}/{concept_counts.mean():.1f}")

# Concepts with fewer than 5 items
thin_concepts = concept_counts[concept_counts < 5]
if len(thin_concepts) > 0:
    print(f"\n  !! Thin concepts (< 5 items): {len(thin_concepts)}")
    for c, n in thin_concepts.items():
        print(f"    {c}: {n} items")

# 4b. a_vector loading diagnostics
print(f"\n  a_vector loading analysis:")
a_nonzero_dims = []
for idx, row in df.iterrows():
    a_vec = ast.literal_eval(row['a_vector'])
    nonzero = [i for i, v in enumerate(a_vec) if v != 0]
    a_nonzero_dims.extend(nonzero)

dim_counts = Counter(a_nonzero_dims)
active_dims = sorted(dim_counts.keys())
print(f"  Active dimensions: {len(active_dims)}/58")
print(f"  Inactive dimensions: {[d for d in range(58) if d not in dim_counts]}")

# 4c. z-vector activation diagnostics (after fixes)
print(f"\n  z-vector activation analysis (post-fix):")
z_dim_counts = Counter()
for idx, row in df.iterrows():
    correct_opt = int(row['correct_option'])
    for opt in range(1, 5):
        if opt == correct_opt:
            continue
        z_vec = parse_z(row[f'z{opt}'])
        for dim_i, v in enumerate(z_vec):
            if v != 0:
                z_dim_counts[dim_i] += 1

print(f"  z-dimension activation counts:")
for dim_i in range(15):
    count = z_dim_counts.get(dim_i, 0)
    bar = '#' * (count // 10)
    print(f"    z_{dim_i:02d}: {count:4d} activations  {bar}")

# 4d. Correct option z-vector verification
post_fix_violations = 0
for idx, row in df.iterrows():
    correct_opt = int(row['correct_option'])
    z_vec = parse_z(row[f'z{correct_opt}'])
    if any(v != 0 for v in z_vec):
        post_fix_violations += 1
print(f"\n  Post-fix correct-option z-vector violations: {post_fix_violations}/{N_ITEMS}")

# 4e. d_param distribution
d_vals = df['d_param'].values
print(f"\n  d_param statistics:")
print(f"    min={d_vals.min():.2f}, max={d_vals.max():.2f}, mean={d_vals.mean():.2f}, std={d_vals.std():.2f}")

# 4f. Prerequisite coverage
prereq_counts = df['prereqs'].value_counts()
no_prereq = (df['prereqs'] == 'None (Foundation Concept)').sum()
has_prereq = N_ITEMS - no_prereq
print(f"\n  Prerequisite coverage:")
print(f"    Foundation items (no prereq): {no_prereq}")
print(f"    Items with prerequisites: {has_prereq}")


# ===========================================================================
# SAVE
# ===========================================================================
print("\n" + "=" * 70)
df.to_csv(OUTPUT_CSV, index=False)
print(f"  OK Saved calibrated Q-matrix to: {OUTPUT_CSV}")
print(f"    {N_ITEMS} items × {len(df.columns)} columns")

# Final integrity checks
print("\n  Final Integrity Checks:")
df_check = pd.read_csv(OUTPUT_CSV)
assert len(df_check) == N_ITEMS, f"Row count mismatch: {len(df_check)} != {N_ITEMS}"
print(f"    OK Row count: {len(df_check)}")

# Verify correct option z-vectors are zero
final_violations = 0
for idx, row in df_check.iterrows():
    correct_opt = int(row['correct_option'])
    z_vec = parse_z(row[f'z{correct_opt}'])
    if any(v != 0 for v in z_vec):
        final_violations += 1
assert final_violations == 0, f"Z-vector fix failed: {final_violations} violations remain"
print(f"    OK All correct-option z-vectors are zero")

# Verify at least one wrong option per item has a non-zero z-vector
items_with_no_traps = 0
for idx, row in df_check.iterrows():
    correct_opt = int(row['correct_option'])
    has_trap = False
    for opt in range(1, 5):
        if opt == correct_opt:
            continue
        z_vec = parse_z(row[f'z{opt}'])
        if any(v != 0 for v in z_vec):
            has_trap = True
            break
    if not has_trap:
        items_with_no_traps += 1
print(f"    OK Items with at least one misconception trap: {N_ITEMS - items_with_no_traps}/{N_ITEMS}")
if items_with_no_traps > 0:
    print(f"    !! {items_with_no_traps} items have no traps (all wrong options slipped)")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("=" * 70)
