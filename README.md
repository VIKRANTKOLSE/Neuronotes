# Neuronotes: Adaptive Testing & Psychometric Simulation Framework

An advanced, research-grade Computerized Adaptive Testing (CAT) and psychometric simulation platform implementing **SMD-CC-MIRT-KL-CAT** (Stochastic Meta-Descent Constrained Multidimensional Item Response Theory with Kullback-Leibler Information Gain and Semantic Misconception Diagnostics).

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture & Core Concepts](#architecture--core-concepts)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Setup & Installation](#environment-setup--installation)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
  - [1. Quick Start (Fast Run)](#1-quick-start-fast-run)
  - [2. Running Experiment Suites](#2-running-experiment-suites)
  - [3. Running Specific Models (--models)](#3-running-specific-models---models)
  - [4. Customizing Learner Pool Size](#4-customizing-learner-pool-size)
  - [5. Skipping Data Processing](#5-skipping-data-processing)
- [Model Catalog & Experimental Stages](#model-catalog--experimental-stages)
- [Output Artifacts & Results](#output-artifacts--results)
- [Testing & Quality Assurance](#testing--quality-assurance)

---

## 🔬 Overview

Neuronotes unifies multi-dimensional latent ability estimation ($\boldsymbol{\theta}$), ontological prerequisite constraints, distractor-level semantic misconception tracking, and adaptive intervention routing.

### Key Capabilities:
- **CC-MIRT (Concept-Constrained MIRT)**: Multidimensional IRT with directed prerequisite graph penalties.
- **C-Matrix Diagnostic Ontology**: Distractor-level misconception mapping using 15-dimensional $z$-vectors.
- **SMD-VSNLMS Updater**: Stochastic Meta-Descent Variable-Step Normalized LMS for semantic-weighted parameter updates.
- **KL-CAT Item Selection**: Dual-objective item selection balancing latent ability information gain and distractor diagnostic entropy with repetition penalties.
- **Dynamic $c_j$ Estimation**: Bayesian update of pseudo-guessing parameters based on distractor entrapment and empirical responses.
- **Intervention Router**: Closed-loop routing of remedial questions (`PREREQ_Q`) and explanations upon detected misconception patterns.

---

## 📁 Project Structure

```text
Neuronotes/
├── data/                               # Clean dataset artefacts
│   ├── items_clean.csv                 # Parsed item bank with psychometric parameters
│   ├── item_options.csv                # Distractor options with misconception tags
│   └── concept_graph.csv               # Directed prerequisite graph edges
├── baselines/                          # Comparison baseline models
│   ├── b1_random.py                    # B1: Random item selection
│   ├── b2_staircase.py                 # B2: 1-up / 1-down staircase CAT
│   ├── b3_3pl_irt.py                   # B3: 1D 3PL-IRT with Fisher information
│   ├── b4_mirt.py                      # B4: Standard Multidimensional IRT
│   └── b5_kl_mirt_no_misc.py           # B5: KL-MIRT without misconception diagnostics
├── neuronotes/                         # Core Neuronotes engine
│   ├── cc_mirt.py                      # Constrained MIRT & Fisher information
│   ├── c_matrix.py                     # Distractor misconception mapping
│   ├── dynamic_c.py                    # Dynamic guessing parameter adaptation
│   ├── kl_cat.py                       # Kullback-Leibler adaptive selector
│   ├── smd_vsnlms.py                   # SMD-VSNLMS state updater
│   ├── intervention_router.py          # Remediation & feedback router
│   └── t_matrix.py                     # Cold-start prior matrix router
├── simulation/                         # Simulation harness & metrics
│   ├── learner_generator.py            # Synthetic learner generation & response simulation
│   ├── run_experiment.py               # Master experiment runner across all stages
│   └── metrics.py                      # RMSE, MAE, ECE, F1, AUROC & Efficiency metrics
├── results/                            # Generated evaluation outputs & figures
├── data_processing.py                  # Ingestion & preprocessing pipeline
├── main.py                             # Master CLI entry point
├── requirements.txt                    # Project Python dependencies
├── setup_venv.bat                      # Windows CMD venv setup script
├── setup_venv.ps1                      # Windows PowerShell venv setup script
└── README.md                           # Documentation
```

---

## ⚙️ Prerequisites

- **Python**: Version 3.10, 3.11, or 3.12.
- **Operating System**: Windows, Linux, or macOS.
- **Package Manager**: `pip` (included with standard Python).

---

## 🚀 Environment Setup & Installation

You can set up your virtual environment automatically using convenience scripts or manually.

### Option A: Automated Convenience Scripts (Windows)

#### Using PowerShell:
```powershell
.\setup_venv.ps1
```
*(If prompted by script execution policies, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

#### Using Command Prompt (CMD):
```cmd
setup_venv.bat
```

---

### Option B: Manual Setup (All Platforms)

#### 1. Create a Virtual Environment
```bash
# Windows
python -m venv .venv

# Linux / macOS
python3 -m venv .venv
```

#### 2. Activate the Virtual Environment
```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux / macOS (Bash / Zsh)
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Step-by-Step Execution Guide

All experiments are driven through the unified CLI entry point `main.py`.

### 1. Quick Start (Fast Run)
Runs the production **P7 Full Neuronotes** model with default parameters:
```bash
python main.py
```

---

### 2. Running Experiment Suites

Use the `--suite` flag to run predefined model comparisons:

- **`final`** *(Default)*: Runs only `P7_Full_Neuronotes`.
  ```bash
  python main.py --suite final
  ```

- **`paper`**: Primary comparison set against standard MIRT and one-factor ablations:
  - `B4_MIRT`, `P7_Full_Neuronotes`, `C1_NoPrereq`, `C2_NoCMatrix`, `C3_FixedC`, `C4_NoSMD`, `C5_NoRouter`
  ```bash
  python main.py --suite paper
  ```

- **`full`**: Runs all 24 models across Baselines, Progressive components, Ablations, Selection strategies, and Feedback policies:
  ```bash
  python main.py --suite full
  ```

---

### 3. Running Specific Models (`--models`)

You can cherry-pick any subset of models for targeted comparison:

```bash
# Compare standard MIRT baseline vs Full Neuronotes
python main.py --models B4_MIRT P7_Full_Neuronotes

# Compare KL selection variations
python main.py --models D4_KL_Only D5_KL_Prereq D6_KL_Misc D7_KL_Full_Pen
```

---

### 4. Customizing Learner Pool Size

Use the `--learners` flag to adjust the number of simulated synthetic learners:

```bash
# Rapid test run with 20 learners
python main.py --learners 20 --suite paper

# Standard experiment run with 200 learners
python main.py --learners 200 --suite paper
```

---

### 5. Multi-Seed Statistical Evaluation (`--seeds` / `--n-seeds`)

Run experiments across multiple random seeds for rigorous statistical aggregation (mean ± std, 95% confidence intervals, and paired difference tests):

```bash
# Run across specific seeds
python main.py --suite paper --learners 200 --seeds 42 43 44 45 46

# Run N consecutive seeds starting from seed 42
python main.py --suite paper --learners 200 --n-seeds 5

# Scaled single seed evaluation
python main.py --suite paper --learners 1000 --seed 42
```

---

### 6. Skipping Data Processing

By default, `main.py` regenerates clean data files in `data/`. When iterating quickly on model code, use `--skip-data` to reuse existing processed datasets:

```bash
python main.py --skip-data --models B4_MIRT P7_Full_Neuronotes
```


---

## 📊 Model Catalog & Experimental Stages

| Category | Model Name | Description |
| :--- | :--- | :--- |
| **Stage A: Baselines** | `B1_Random` | Random item selection baseline |
| | `B2_Staircase` | 1-up / 1-down adaptive staircase |
| | `B3_3PL_IRT` | 1D 3PL IRT with Fisher Information item selection |
| | `B4_MIRT` | Standard 3-dimensional MIRT with Max-Fisher selection |
| | `B5_KL_NoMisc` | Multidimensional KL item selection (no misconception tracking) |
| **Stage B: Progressive** | `P1_MIRT_KL` | Unconstrained MIRT with KL item selection |
| | `P2_CCMIRT_KL` | Concept-Constrained MIRT + KL selection |
| | `P3_CCMIRT_CMatrix` | CC-MIRT + C-Matrix distractor diagnosis |
| | `P4_DynC` | CC-MIRT + C-Matrix + Dynamic $c_j$ guessing parameter adaptation |
| | `P5_SMD` | CC-MIRT + C-Matrix + Dynamic $c_j$ + SMD-VSNLMS adaptive updates |
| | `P6_Router` | CC-MIRT + C-Matrix + Dyn $c_j$ + SMD + Remedial intervention routing |
| | `P7_Full_Neuronotes`| **Complete Full Neuronotes pipeline** |
| **Stage C: Ablations** | `C1_NoPrereq` | Full pipeline without prerequisite graph constraints |
| | `C2_NoCMatrix` | Full pipeline without C-Matrix distractor diagnosis |
| | `C3_FixedC` | Full pipeline with fixed guessing parameter ($c_j = 0.25$) |
| | `C4_NoSMD` | Full pipeline without SMD-VSNLMS (standard gradient descent) |
| | `C5_NoRouter` | Full pipeline without intervention routing |
| **Stage D: Selection** | `D1_Random_FullPsy`| Random selection with full Neuronotes psychometric tracking |
| | `D3_MaxFisher` | Maximum Fisher Information selection |
| | `D4_KL_Only` | KL divergence selection without prerequisite penalties |
| | `D5_KL_Prereq` | KL divergence with prerequisite weighting |
| | `D6_KL_Misc` | KL divergence focused on misconception entropy |
| | `D7_KL_Full_Pen` | Full KL selection with repetition & exposure penalties |
| **Stage E: Feedback** | `E1_NoFeedback` | Full pipeline without feedback delivery |
| | `E2_GenericFeedback`| Generic correctness feedback |
| | `E3_ConceptFeedback`| Concept-level corrective feedback |
| | `E4_MiscFeedback` | Distractor-level misconception rationale feedback |
| | `E5_MiscPlusPrerFeedback`| Misconception feedback with prerequisite remedial routing |

---

## 📈 Output Artifacts & Results

All outputs are saved to the `results/` directory:

1. **`simulation_results.csv`**: Comprehensive metrics table:
   - Ability Estimation: `rmse`, `rmse_std`, `rmse_ci95_lo`, `rmse_ci95_hi`, `mae`
   - Calibration Quality: `ece` (Expected Calibration Error)
   - Misconception Classification: `precision`, `recall`, `f1`, `auroc` (macro), `auroc_micro`
   - Test Efficiency: `efficiency_q` (mean questions to reach $\text{RMSE} \le 0.35$)
2. **`rmse_trajectory.csv`**: Convergence path of mean $\theta$ estimation error per test step for every evaluated system.
3. **`calibrated_item_bank.csv`**: Item bank parameters and exposure statistics.
4. **`metric_comparison.png`**: Multi-panel visualization comparing RMSE, F1 score, AUROC, and Efficiency across evaluated models.
5. **`rmse_comparison.png`**: Convergence trajectory comparison plot showing question efficiency towards the RMSE $\le 0.35$ threshold.

---

## 🧪 Testing & Quality Assurance

To verify that all files compile and run properly:

```bash
# Verify Python syntax across all modules
python -c "import py_compile, glob; [py_compile.compile(f, doraise=True) for f in glob.glob('**/*.py', recursive=True)]; print('All modules compiled successfully!')"

# Run a fast smoke test
python main.py --skip-data --learners 10 --models B4_MIRT P7_Full_Neuronotes
```
