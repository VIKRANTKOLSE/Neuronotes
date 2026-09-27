# Neuronotes Agent Reference — SMD-CC-MIRT-KL-CAT Codebase

> **Purpose**: This file is the single-source reference for any agent working in this repository.
> Read this FIRST to avoid wasting tokens reading every file.

---

## 1. Project Overview

**Neuronotes** is a psychometric adaptive testing system for **Inorganic Chemistry** (JEE Advanced level).
It implements the **SMD-CC-MIRT-KL-CAT** pipeline:

- **SMD-VSNLMS** — Semantic variable-step normalized LMS theta updates (58D ability space)
- **CC-MIRT** — Chemically Constrained Multidimensional IRT (58D ability model across 4 concept tiers)
- **KL-CAT** — KL-divergence-based Computerized Adaptive Testing item selector
- **C-Matrix** — Option-level confusion/misconception matrix (15-dim shared ontology)
- **Dynamic-C** — Adaptive guessing floor parameters
- **Intervention Router** — Rule-based pedagogical feedback routing

The system simulates learners, runs adaptive tests, and evaluates across 28 model configurations (baselines, progressive builds, ablations, selection strategies, feedback policies).

---

## 2. Directory Structure

```
codes/
├── data_processing.py          # Step 1: raw CSV → 3 clean artefacts
├── main.py                     # Master experiment pipeline (CLI entry point)
├── mirt_vs_uirt_sim.py         # Standalone MIRT vs UIRT Monte Carlo comparison
├── requirements.txt            # numpy, pandas, scipy, scikit-learn, networkx, matplotlib
├── setup_venv.bat / .ps1       # Virtual environment setup scripts
│
├── data/                       # Generated artefacts & source data
│   ├── questions_final_qmatrix.csv # 548 items with 58D a_vectors
│   ├── items_clean.csv         # 548 items, parsed item bank (58D a_vector as JSON)
│   ├── item_options.csv        # 2192 rows (4 options × 548 items)
│   └── concept_graph.csv       # DAG edges (prerequisite relationships)
│
├── neuronotes/                 # Core package (6 modules)
│   ├── __init__.py             # Exports: CCMIRT, CMatrix, DynamicC, SMDVSNLMSUpdater, KLCAT, InterventionRouter
│   ├── cc_mirt.py              # Module 1: 3PL MIRT + prerequisite graph constraints (58D)
│   ├── c_matrix.py             # Module 2: Option → misconception tag mapping (15D ontology)
│   ├── dynamic_c.py            # Module 3: Adaptive guessing floor (c_j)
│   ├── smd_vsnlms.py           # Module 4: Theta update engine (58D theta, 58x15 B matrix)
│   ├── kl_cat.py               # Module 5: Item selection (58D KL gain + prereq + misc + exposure)
│   ├── t_matrix.py             # Cold-start routing over prerequisite DAG
│   └── intervention_router.py  # Module 6: Pedagogical intervention rules
│
├── baselines/                  # 5 baseline selectors
│   ├── __init__.py
│   ├── b1_random.py            # Random item selection
│   ├── b2_staircase.py         # Up-down staircase method
│   ├── b3_3pl_irt.py           # Standard 3PL IRT (1D Fisher info with MDISC scalar)
│   ├── b4_mirt.py              # MIRT (58D trace Fisher info)
│   └── b5_kl_mirt_no_misc.py   # KL-MIRT without misconception tracking (58D)
│
├── simulation/                 # Experiment harness
│   ├── __init__.py
│   ├── run_experiment.py       # System factory + adaptive test loop (28 configs)
│   ├── learner_generator.py    # Synthetic learner population (58D profiles across 4 tiers)
│   └── metrics.py              # RMSE, MAE, ECE, F1, AUROC, efficiency, Cohen's d
│
└── results/                    # Output artefacts
    ├── simulation_results.csv
    ├── multi_seed_results.csv
    ├── rmse_trajectory.csv
    ├── calibrated_item_bank.csv
    ├── metric_comparison.png
    └── rmse_comparison.png
```

---

## 3. Data Pipeline

### 3.1 Raw Data Source

- **Source**: `data/questions_final_qmatrix.csv`
- **548 items**, each with 4 MCQ options, 58-dimensional `a_vector` (Q-matrix concept node loadings), and 15-dimensional distractor `z`-vectors.

### 3.2 Raw CSV Columns

| Column | Type | Description |
|---|---|---|
| `source_id` | str | Unique item identifier (e.g., `EffNucCharge_01`) |
| `question_text` | str | Stem of question |
| `opt1`–`opt4` | str | Option text |
| `reason1`–`reason4` | str | Rationale/explanation for each option |
| `concept` | str | Chemistry concept name |
| `prereqs` | str | Prerequisite concept names |
| `a_vector` | str (list) | 58-dim item discrimination vector |
| `d_param` | float | Difficulty parameter |
| `correct_option` | int | Correct option number (1-4) |
| `z1`–`z4` | str (list) | 15-dim misconception z-vector per option |

### 3.3 Generated Artefacts

#### `data/items_clean.csv` (548 rows)

| Key Columns | Source |
|---|---|
| `item_id` | From raw `source_id` |
| `concept` | Direct from raw |
| `prereqs` | Direct from raw |
| `correct_option` | Direct (1–4) |
| `a_vector` | JSON string representing 58-dimensional numpy array |
| `d_param` | Direct float |
| `entrapment_index` | Computed: mean `trap_weight` of wrong options |
| `semantic_entrapment` | Same as `entrapment_index` |
| `c_j` | Computed via `initial_dynamic_c()` |
| `empirical_entrapment` | Initialized to 0.0 (updated at runtime) |
| `ambiguity_index` | Initialized to 0.0 (updated at runtime) |
| Metadata defaults | `estimated_time_sec` (90), `item_exposure_limit` (60), `status` ("active"), etc. |

#### `data/item_options.csv` (2192 rows = 4 x 548)

| Column | Source |
|---|---|
| `item_id` | From raw `source_id` |
| `option_no` | 1–4 |
| `option_text` | From raw `opt{N}` |
| `is_correct` | `option_no == correct_option` |
| `rationale` | From raw `reason{N}` |
| `misconception_tag` | **Derived**: `infer_misconception_tag(z_vec)` — first active z-dim as `z_XX` |
| `error_class` | **Derived**: `infer_error_class(reason)` — keyword heuristic on reason text |
| `severity` | **Derived**: `infer_severity(error_class, z_vec)` — count of active z-dims |
| `trap_weight` | **Derived**: `sum(z_vec) / 15.0` |
| `semantic_dimension` | **Derived**: `infer_semantic_dimension(a_vector)` — coarse cognitive category |
| `expert_confidence` | Hardcoded `"high"` |
| `review_status` | `"active"` |
| `z_vector` | Stringified `z{N}` list (15 dimensions) |

#### `data/concept_graph.csv`

Directed prerequisite edges across 58 chemistry concepts. Built from `CONCEPT_PREREQ_MAP` dict in `data_processing.py`.
Columns: `source_concept`, `target_concept`, `relation`, `weight`, `expert_validated`.

---

## 4. Core Modules (neuronotes/)

### 4.1 `cc_mirt.py` — Chemically Constrained MIRT (Module 1)

- **Class**: `CCMIRT`
- **Ability model**: 58-dimensional ability vector $\theta$ (one dimension per concept node in DAG).
  - Tier 1: 10 concepts (indices 0–9) — Foundational atomic structure
  - Tier 2: 13 concepts (indices 10–22) — Periodic trends & bonding principles
  - Tier 3: 17 concepts (indices 23–39) — Advanced bonding & d/f-block
  - Tier 4: 18 concepts (indices 40–57) — Coordination chemistry & metallurgy
- **Response probability**: Compensatory MIRT model:
  $$P = c_j + \frac{1 - c_j}{1 + \exp(-\mathbf{a}_j^T \boldsymbol{\theta} - d_j)}$$
- **Prerequisite soft constraint**: If concept $B$ requires prerequisite $A$, soft loss penalizes $\theta_B > \theta_A + \text{slack}$.
- **Fisher information**: $58 \times 58$ matrix.
- **CONCEPT_DIM_MAP**: Maps each of the 58 concepts to its primary Q-matrix dimension (0–57).
- **Key params**: `PREREQ_SLACK = 0.5`, `N_DIMS = 58`.
- **Helper**: `parse_a_vector(value)` parses JSON/list/array into a 58D numpy array.

### 4.2 `c_matrix.py` — Confusion Matrix (Module 2)

- **Class**: `CMatrix`
- **Maps**: `(item_id, option_no)` to misconception profile dict.
- **Loads from**: `data/item_options.csv`.
- **MISCONCEPTION_ONTOLOGY**: 15 shared z-dimensions (`z_00` to `z_14`), each a named chemistry misconception.
- **Key methods**: `diagnose()`, `is_correct()`, `z_vector()`, `all_wrong_options()`, `misconception_diagnostic_value()`.

### 4.3 `dynamic_c.py` — Dynamic Guessing Floor (Module 3)

- **Class**: `DynamicC`
- **Formula**: $c_j = 0.25 \cdot \sigma(\beta_0 + \beta_{\text{sem}} E_{\text{sem}} + \beta_{\text{emp}} E_{\text{emp}} - \beta_{\text{amb}} E_{\text{amb}})$
- **Loads from**: `data/items_clean.csv`.
- **Recalibration**: Every $N$ responses (default 100).
- **Factory**: `DynamicC.fixed(0.25)` for ablation.

### 4.4 `smd_vsnlms.py` — Semantic Update Engine (Module 4)

- **Class**: `SMDVSNLMSUpdater`
- **State class**: `LearnerState` ($\boldsymbol{\theta} \in \mathbb{R}^{58}$, `concept_theta`, `misconception`, `misconception_probs` $\in \mathbb{R}^{15}$, `repeat_count`, `response_history`, `previous_delta` $\in \mathbb{R}^{58}$).
- **Update formula**:
  $$\Delta \boldsymbol{\theta} = \eta \cdot (\text{residual} \cdot \mathbf{a}_j + \alpha \mathbf{B} \mathbf{z}_{jk}) + \mu \Delta_{\text{prev}}$$
  - $\mathbf{B}$: $58 \times 15$ learnable semantic projection matrix updated online from residuals.
  - $\eta$: adaptive learning rate scaled by error class and repetition.
  - $\mathbf{z}$: 15-dim misconception vector from C-Matrix.

### 4.5 `kl_cat.py` — KL-CAT Item Selector (Module 5)

- **Class**: `KLCAT`
- **Selection criterion**:
  $$U_j = w_{\text{kl}} E_{\text{KL}}(j) + w_{\text{pre}} \text{prereq\_score}(j) + w_{\text{mis}} \text{diag\_value}(j) - w_{\text{rep}} \text{exposure\_penalty}(j)$$
- **Default weights**: KL=0.50, Prereq=0.20, Misc=0.20, Rep=0.10.
- **Performance**: Pre-caches parsed $\mathbf{A} \in \mathbb{R}^{N \times 58}$ numpy matrix for vectorized expected KL gain calculation.

### 4.6 `t_matrix.py` — Cold-Start Router

- **Class**: `ColdStartRouter`
- **Strategy**: Recommends foundation concepts based on prerequisite DAG topology before sufficient response evidence.

### 4.7 `intervention_router.py` — Intervention Rules (Module 6)

- **Class**: `InterventionRouter`
- **Intervention types**: NORMAL_Q, HINT, EXPLANATION, PREREQ_Q, CONTRAST_Q, REMEDIAL_SET.
- **Routing**: Uses `CONCEPT_DIM_MAP` to evaluate concept-specific ability $\theta[\text{dim}]$.

---

## 5. Simulation Framework

### 5.1 `simulation/learner_generator.py`

- **Class**: `LearnerGenerator`
- **6 learner archetypes**: `strong_all`, `weak_all`, `strong_found`, `strong_bonding`, `strong_coord`, `mixed`.
- **Ground truth**: 58-dimensional $\boldsymbol{\theta}_{\text{true}}$ sampled across the 4 concept tiers; $\boldsymbol{\theta}_{\text{init}} = \mathbf{0}_{58}$.
- **Response simulation**: Uses 58D dot product $\mathbf{a}_j^T \boldsymbol{\theta}_{\text{true}} + d_j$.

### 5.2 `simulation/run_experiment.py`

- **28 system configurations** across 5 stages:
  - **Stage A (B1-B5)**: Baselines (Random, Staircase, 3PL IRT, MIRT, KL-NoMisc)
  - **Stage B (P1-P7)**: Progressive Neuronotes build
  - **Stage C (C1-C5)**: Ablations from P7
  - **Stage D (D1-D7)**: Selection strategy comparison
  - **Stage E (E1-E5)**: Feedback policy comparison
- **make_system()**: Factory returning fresh, isolated components.
- **MAX_QUESTIONS = 30** per learner session.

### 5.3 `simulation/metrics.py`

- **MetricBundle**: RMSE (58D Euclidean error), MAE, ECE, Precision, Recall, F1, AUROC (macro & micro over 15 misconception dimensions), efficiency_q.

---

## 6. Baselines (baselines/)

| Baseline | Strategy | 58D Adaptation |
|---|---|---|
| B1_Random | Random item from unseen pool | Fully dimension-agnostic |
| B2_Staircase | Up/down difficulty tracking current mean ability | Uses `np.mean(theta)` |
| B3_3PL_IRT | Max scalar Fisher info (1D) | Derives scalar discrimination via MDISC ($\|\mathbf{a}_j\|_2$) |
| B4_MIRT | Max trace of Fisher info matrix | Computes $\text{trace}(\mathbf{I}) = \text{scale} \cdot \|\mathbf{a}_j\|_2^2$ across 58D |
| B5_KL_NoMisc | KL-MIRT without misconception component | Vectorized 58D posterior divergence |

---

## 7. Key Constants and Parameters

| Constant | Value | Location |
|---|---|---|
| `N_DIMS` | 58 | `cc_mirt.py` |
| `N_SEMANTIC_DIMS` | 15 | `smd_vsnlms.py`, `c_matrix.py` |
| `PREREQ_SLACK` | 0.5 | `cc_mirt.py` |
| `BASE_LR` | 0.15 | `smd_vsnlms.py` |
| `MAX_QUESTIONS` | 30 | `run_experiment.py` |
| `RMSE_THRESHOLD` | 0.35 | `run_experiment.py` |
| `MAX_EXPOSURE` | 60 | `kl_cat.py` |
| `THETA_LOW_THRESH` | -1.0 | `intervention_router.py` |
| KL-CAT weights | 0.50 / 0.20 / 0.20 / 0.10 | `kl_cat.py` |
| Dynamic-C coefficients | b0=2.0, b_sem=-1.6, b_emp=-1.2, b_amb=0.8 | `dynamic_c.py` |

---

## 8. CLI Usage

```bash
py main.py                                        # Fast: P7 only (seed 42)
py main.py --suite paper                           # Paper comparison (7 models)
py main.py --suite full                            # Complete 28-model benchmark
py main.py --models B4_MIRT P7_Full_Neuronotes     # Cherry-pick
py main.py --suite paper --seeds 42 43 44 45 46    # Multi-seed
py main.py --suite paper --learners 1000           # Scale population
py main.py --skip-data                             # Skip data_processing step
```
