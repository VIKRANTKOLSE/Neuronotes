# Neuronotes SMD-CC-MIRT-KL-CAT: Engineering & Mathematical History

This document chronicles the full sequence of mathematical, psychometric, and architectural issues identified, analyzed, and resolved across the codebase.

---

## 1. 3D to 58D Psychometric Vector Refactoring

### Issue Identified
* The legacy codebase operated on 3-dimensional latent traits (`theta` of length 3), which was incompatible with the actual 58-concept curriculum derived from `data/questions_final_qmatrix.csv` (548 items).
* TypeScript/Node frontend prototypes (`src/`) created architectural divergence and redundant maintenance.

### Resolution & Implementation
* **Removed TypeScript/Node files**: Standardized on a pure Python pipeline under `codes/`.
* **58-Dimensional Curriculum Mapping**:
  * Updated [cc_mirt.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/cc_mirt.py) with `N_DIMS = 58` and mapped all 58 curriculum concepts into `CONCEPT_DIM_MAP` based on the Q-matrix loadings.
  * Added `parse_a_vector()` to handle serialized 58D JSON discrimination arrays.
* **Full Pipeline Propagation**:
  * [smd_vsnlms.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/smd_vsnlms.py): Expanded `LearnerState.theta` to 58D and initialized the semantic mapping matrix $B$ with shape $(58, 15)$.
  * [kl_cat.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/kl_cat.py): Pre-cached $(548, 58)$ numpy discrimination matrix $A$ for vectorized expected KL divergence computation in $\mathcal{O}(1)$ time per question.
  * `baselines/` & `simulation/`: Refactored all 24 baseline, progressive, ablation, and selection models to evaluate 58D ability vectors.
  * `simulation/learner_generator.py`: Structured synthetic learner abilities across 4 curriculum tiers (Foundations, Bonding, Energetics, Coordination).

---

## 2. Dynamic Learning Rate ($\eta_s$) for SMD-VSNLMS

### Issue Identified
* In multidimensional CAT, wild guessing or random slips generate erratic psychometric residuals ($r_{jk} = y - P$).
* Standard constant learning rates allowed wild guesses to distort the 58D ability trajectory, corrupting ability estimates.

### Resolution & Implementation
Integrated an adaptive variance EMA step size in [smd_vsnlms.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/smd_vsnlms.py):
1. **Variance EMA**:
   $$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot (r_{jk}^2)$$
2. **Adaptive Step Size**:
   $$\eta_s = \max\left(\eta_{\min}, \frac{\eta_{\max}}{1 + \rho \cdot v_t}\right)$$
* **Hyperparameters**: $\beta = 0.9$, $\eta_{\max} = 0.5$, $\eta_{\min} = 0.05$, $\rho = 10.0$.
* **Unit Testing**: Implemented 6 unit tests in [test_dynamic_learning_rate.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/test_dynamic_learning_rate.py), confirming deterministic convergence and resistance to guessing spikes.

---

## 3. Prerequisite DAG Evidence Propagation

### Issue Identified
* In a 58-concept curriculum, testing one concept provides indirect causal evidence about prerequisites and downstream concepts that traditional MIRT treats as independent.

### Resolution & Implementation
* Implemented graph-theoretic evidence propagation in [cc_mirt.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/cc_mirt.py) across the 81-edge prerequisite DAG:
  * Precomputed all-pairs shortest paths and distance-attenuated causal matrices:
    $$\mathbf{M}_{\text{up}}[src, tgt] = \alpha_{\text{up}} \cdot \gamma^{\text{dist} - 1}$$
    $$\mathbf{M}_{\text{down}}[tgt, src] = \alpha_{\text{down}} \cdot \gamma^{\text{dist} - 1}$$
  * Correct responses propagate positive evidence upward to prerequisite ancestors.
  * Incorrect responses propagate negative evidence downward to downstream descendants.

---

## 4. Root Cause Analysis: The 0.94 RMSE Floor in 30-Question CAT

### Issue Identified
* Despite the dynamic learning rate and DAG propagation, RMSE remained stalled around **0.94** in 30-question sessions.

### Empirical Diagnosis
* **The "Untouched Dimensions" Floor**:
  * With 58 dimensions and 30 questions, KLCAT greedily chose items with high loading norms $\|\mathbf{a}_j\| > 2.0$.
  * KLCAT repeatedly probed only **14 concepts** (e.g. *Bent's Rule* 5 times), leaving **44 concepts (76%) untouched at prior $\theta_k = 0.0$**.
  * Ground truth ability $\theta_{\text{true}}$ had average magnitude $\approx 1.05$.
  * The untouched 44 dimensions alone created an inescapable mathematical MSE floor:
    $$\text{MSE}_{\text{untouched}} = \frac{44}{58} \times \mathbb{E}[\theta_{\text{true}}^2] \approx 0.76 \times 1.10 = 0.836 \implies \mathbf{\text{RMSE} \ge 0.914}$$
* **LMS Step Size Under-Scaling**:
  * In standard LMS, $\Delta \theta_k \approx 0.12$. An algorithm calibrated for 500 gradient steps cannot converge on dimensions tested only once.

---

## 5. Investigation of Gradient Drift at $N=200$ (Robbins-Monro Violation)

### Issue Identified
* When the test length was scaled to 200 questions to give all concepts multiple exposures, RMSE initially dropped to a minimum at Step 52 (0.957), but then **reversed and drifted upwards to 1.0255 at Step 200**.
* **Principle**: In statistical learning, if more data increases estimation error, the update algorithm is mathematically unstable and accumulating biased gradient errors over time.

### Systematic Ablation Results

| Configuration | Step 1 | Step 30 | Step 50 | Step 100 | Step 200 | Trajectory Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full System (Unfixed)** | 1.1162 | 1.0558 | 1.0378 | 1.0113 | **1.0210** | **Unstable**: Drifts up after step 115 |
| **No Soft Prereq Penalty** | 1.1154 | 1.0487 | 1.0287 | 1.0034 | **1.0100** | **Unstable**: Still drifts up |
| **No DAG Propagation** | 1.1155 | 1.0668 | 1.0475 | 1.0193 | **0.9881** | **Monotonically converges** |
| **Pure SMD (No Graph)** | 1.1160 | 1.0672 | 1.0476 | 1.0187 | **0.9865** | **Monotonically converges** |
| **Pure SMD (No Semantic $B$)** | 1.1163 | 1.0818 | 1.0659 | 1.0314 | **0.9713** | **Optimal monotonic convergence** |

### Identified Root Causes
1. **Hard Non-Vanishing Floor in DAG Propagation (`mag = max(0.08, ...)`):**
   * Even when prediction error vanished ($r \to 0$), a minimum perturbation of $\pm 0.08$ was forcibly injected into connected nodes, violating the Robbins-Monro requirement $\sum \gamma_t^2 < \infty$.
2. **Directional Graph Topology Asymmetry:**
   * Foundational concepts had up to 30 descendants; advanced concepts had 0 descendants. Missing a basic question pulled down 30 concepts, but missing an advanced question did not touch ancestors, accumulating persistent downward bias on advanced concepts.
3. **Un-annealed Soft Prerequisite Penalty (`learning_rate = 0.05` constant):**
   * Over 200 steps, constant regularization applied over 10.0 units of penalty force, overpowering data likelihood.
4. **One-Sided Negative Gradient Updates in Semantic Matrix $B$:**
   * $B$ was updated only on incorrect responses where $r < 0$. The update $\Delta B = \eta (r \cdot a) z^T$ was strictly negative, causing $B$ to saturate at $-1.0$ and act as a negative ability sink.

---

## 6. Implementation of Robbins-Monro Convergence Fixes

### Code Modifications
1. **Annealed DAG Propagation Magnitude** in [cc_mirt.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/cc_mirt.py):
   ```python
   def calculate_dag_magnitude(residual: float, t: int, gamma_0: float = 0.35, beta: float = 1.0) -> float:
       gamma_t = gamma_0 / math.sqrt(1.0 + beta * t)
       return min(gamma_t * abs(residual), 0.35)
   ```
   * Removed the hard `0.08` lower bound. Perturbations vanish as $t \to \infty$ and $r \to 0$.
2. **Decayed Prerequisite Regularization Rate** in [cc_mirt.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/cc_mirt.py) & [run_experiment.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/simulation/run_experiment.py):
   ```python
   eta_prereq = learning_rate / (1.0 + 0.02 * t)
   theta[dim] -= eta_prereq * gradient
   ```
3. **Centered Symmetric Semantic Matrix $B$ Updates** in [smd_vsnlms.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/smd_vsnlms.py):
   ```python
   if np.any(z):
       self.B += self.semantic_matrix_lr * np.outer(residual * a_vec, z)
       self.B = np.clip(self.B, -1.0, 1.0)
   ```
   * Updates on both correct ($y=1$) and incorrect ($y=0$) responses using centered residual $(y - P)$.

### Resulting Convergence Profile
* Final RMSE dropped from **1.0255 $\to$ 0.8972**.
* Expected Calibration Error (ECE) plummeted from **0.1231 $\to$ 0.0392** (70% error reduction).
* Trajectory confirmed **strict monotonic convergence**:
  ```
  Step   1: 1.0113
  Step  30: 0.9651
  Step  50: 0.9463
  Step 100: 0.9180
  Step 150: 0.9026
  Step 200: 0.8973
  ```

---

## 7. Simulation Runtime Optimization

### Issue Identified
* Running 200 questions across 200 learners (40,000 steps) was stalling and taking >30 minutes due to two internal bottlenecks:
  1. `candidates.iterrows()` in `kl_cat.py` executing 400 row iterations per question whenever an active misconception intervention occurred.
  2. `pd.DataFrame(self._prereq_attempt_log)` re-instantiating 40,000-row DataFrames every 100 questions in `cc_mirt.py`.

### Resolution & Implementation
* Precomputed `self._items_with_misc: dict[str, set[str]]` in [kl_cat.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/kl_cat.py) for $\mathcal{O}(1)$ active misconception candidate boosting:
  ```python
  boost = np.isin(cand_ids, list(target_items)).astype(float) * 0.5
  ```
* Pre-cached static numpy arrays (`_A_cache`, `_d_params`, `_c_params`, `_concept_dims`, `_exposure_limits`).
* Converted prerequisite accuracy tracking to online dict counters (`_concept_correct_counts` and `_concept_total_counts`).
* **Runtime dropped from >30 minutes to ~2 minutes** for the complete 40,000-step benchmark.

---

## 8. Targeted Misconception Decay ("Global Memory Wipe" Fix)

### Issue Identified
* Following the convergence fixes, diagnostic AUROC showed a slight drop from 0.7079 to 0.6807.
* **Root Cause Found**: In `smd_vsnlms.py`, on every correct response:
  ```python
  if correct:
      state.misconception_probs *= (1.0 - MISC_DECAY)  # MISC_DECAY = 0.05
  ```
* Over 200 questions, students answered ~122 questions correctly.
  $$(1 - 0.05)^{122} = 0.95^{122} \approx 0.0019$$
* Every correct response **globally wiped all 15 misconception dimensions**, even for completely unrelated concepts, collapsing probabilities to the $0.001$ floor and degrading classification ranking resolution.

### Resolution & Implementation
1. **Targeted Distractor Masking** in [smd_vsnlms.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/smd_vsnlms.py):
   * Correct answers now **only decay misconceptions that were present as distractor traps in the question**:
     ```python
     if correct:
         N = total_session_length if total_session_length is not None else getattr(self, "total_session_length", 30)
         lambda_decay = min(0.05, 1.5 / max(int(N), 1))
         if z_mask is not None:
             mask = np.asarray(z_mask, dtype=float).reshape(-1)
             mask = np.pad(mask[:N_SEMANTIC_DIMS], (0, max(0, N_SEMANTIC_DIMS - len(mask))))
         else:
             mask = np.ones(N_SEMANTIC_DIMS, dtype=float)
         state.misconception_probs *= (1.0 - lambda_decay * mask)
     ```
2. **Adaptive Test-Length Decay Rate**:
   * Scaled decay rate: $\lambda_{\text{decay}} = \min(0.05, 1.5 / N)$.
   * For $N=30$, $\lambda = 0.05$. For $N=200$, $\lambda = 0.0075$.
3. **Distractor Mask Extraction** in [c_matrix.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/c_matrix.py):
   * Added `item_distractor_mask(item_id, correct_option)` returning the 15D binary mask of traps active among wrong options.
4. **Unit Testing**:
   * Added `TestTargetedMisconceptionDecay` in [test_dynamic_learning_rate.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/test_dynamic_learning_rate.py).
   * Verified selective decay, length adaptation, and array broadcasting with zero shape mismatches (9/9 tests pass).

---

## 9. Final Benchmark Comparison (200 Learners × 200 Questions)

| Benchmark State | Macro AUROC | Micro AUROC | F1 Score | Precision | Final RMSE | ECE | Convergence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Initial Drift State (Unfixed)** | 0.7079 | 0.6974 | 0.3386 | 0.3204 | 1.0255 | 0.1231 | **Drifting** after step 52 |
| **Robbins-Monro Stabilized** | 0.6807 | 0.6728 | 0.2920 | 0.3887 | 0.8972 | 0.0392 | **Strictly Monotonic** |
| **With Targeted Misconception Decay** | **0.7027** | **0.6962** | **0.3656** | **0.5264** | **0.8972** | **0.0392** | **Strictly Monotonic** |

### Key Improvements Achieved:
1. **Precision**: Jumped from $0.3887 \to \mathbf{0.5264}$ (+35.4%).
2. **F1 Score**: Jumped from $0.2920 \to \mathbf{0.3656}$ (+25.2%).
3. **AUROC**: Surpassed the 0.70 threshold to $\mathbf{0.7027}$ (macro) and $\mathbf{0.6962}$ (micro).
4. **Calibration (ECE)**: Maintained at $\mathbf{0.0392}$ (70% error reduction vs baseline).
5. **Convergence**: Strict, non-divergent monotonic RMSE reduction throughout all 200 steps.

---

## 10. Q-Matrix Dataset Calibration & Rectification

### Issues Identified in `questions_final_qmatrix.csv`
1. **100% Violation Rate on Correct Option z-vectors**: All 548 items had non-zero misconception $z$-vectors on the correct option, feeding false misconception signals on every correct answer.
2. **Corrupted Math Symbols**: 83 instances of `?` placeholders replacing Greek symbols ($\sigma$, $\Delta$, $\Delta G^\circ$, $\Delta H^\circ$, $\Delta S^\circ$, $\approx$).
3. **Absence of Slip Noise**: 100% deterministic distractor mappings uncharacteristic of real testing.

### Rectification Applied (`rectify_qmatrix.py` → `questions_calibrated_qmatrix.csv`)
1. **FIX 1**: Zeroed all correct-option $z$-vectors across 548 items.
2. **FIX 2**: Replaced corrupted math symbols with proper UTF-8 Greek symbols.
3. **FIX 3**: Injected 15% procedural slip noise (zeroed 247/1644 wrong-option distractor vectors, ensuring at least one trap remained per item).
4. **FIX 4**: Comprehensive 58-concept coverage diagnostic (all 58 dims active, 56 unique concepts, minimum 3 items per concept).

---

## 11. 200-Question Paper Suite Benchmark & Root-Cause Diagnosis

### Empirical Results ($N = 200$ questions, 200 learners, seed 42)

| System | RMSE | MAE | ECE | F1 | Macro AUROC | Trajectory Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **B4_MIRT** | **0.8811** | 0.7745 | 0.0486 | 0.2320 | 0.5000 | Monotonic convergence |
| **P7_Full_Neuronotes** | 1.0048 | 0.8687 | 0.1312 | **0.3418** | **0.6957** | Min at step 74 (0.9470), drifts to 1.0048 |
| **C1_NoPrereq** | 1.0071 | 0.8611 | 0.1499 | 0.3046 | **0.7307** | Min at step 74 (0.9580), drifts to 1.0071 |
| **C2_NoCMatrix** | **0.8601** | **0.7483** | **0.0350** | 0.2320 | 0.5000 | Optimal monotonic convergence |
| **C3_FixedC** | 1.0031 | 0.8656 | 0.1285 | **0.3609** | **0.7143** | Min at step 74 (0.9430), drifts to 1.0031 |
| **C4_NoSMD** | **0.9014** | 0.7899 | 0.0427 | 0.2320 | 0.5000 | Monotonic convergence |
| **C5_NoRouter** | 1.0091 | 0.8720 | 0.1292 | 0.3375 | 0.7075 | Min at step 74 (0.9445), drifts to 1.0091 |

### Diagnostic Findings
1. **Diagnostic AUROC scales up to 0.7307**: Scaling from 30 to 200 questions increased diagnostic AUROC (+0.07) and F1 (+0.11 to 0.36), validating the calibrated dataset.
2. **Cross-Learner Semantic Matrix Accumulation**: `updater.B` was shared across all 200 learners without reset in `run_experiment.py`.
3. **One-Sided Negative Gradient on $\mathbf{B}$**: Since FIX 1 zeroed correct-option $z$-vectors, `np.any(z)` is only true on errors, depriving $\mathbf{B}$ of positive gradients and creating an artificial negative ability sink over long sessions.

---

## 12. Resolution of Semantic Matrix Sink & Cross-Learner Leakage

### Mathematical Fixes Implemented
1. **Per-Learner State Isolation (`updater.reset()`)** in [smd_vsnlms.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/smd_vsnlms.py) & [run_experiment.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/simulation/run_experiment.py):
   * Added `reset()` to `SMDVSNLMSUpdater` restoring $\mathbf{B}$ to its initial prior.
   * `run_one_learner` invokes `updater.reset()` at the start of every learner session, eliminating cross-learner error propagation.
2. **Symmetric Dual-Phase Gradient Updates on $\mathbf{B}$** in [smd_vsnlms.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/smd_vsnlms.py):
   * On **correct** responses ($r = 1 - P > 0$): Learner successfully rejected the distractor traps on that item. Updated with avoided traps $\mathbf{z}_{\text{mask}}$:
     $$\Delta \mathbf{B} = +\eta_B (r \cdot \mathbf{a}) \mathbf{z}_{\text{mask}}^T$$
   * On **incorrect** responses ($r = -P < 0$): Learner fell for chosen distractor $\mathbf{z}$:
     $$\Delta \mathbf{B} = +\eta_B (r \cdot \mathbf{a}) \mathbf{z}^T$$
   * Restores zero-centered, unbiased equilibrium to $\mathbf{B}$ throughout testing.
3. **Robbins-Monro Semantic Alpha Annealing** in [smd_vsnlms.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/neuronotes/smd_vsnlms.py):
   * Anneals semantic feedback force $\alpha_{\text{semantic}}$ with question step $t$:
     $$\alpha_t = \frac{\alpha_0}{\sqrt{1 + 0.05 \cdot t}}$$
   * Ensures asymptotic stability: strong diagnostic influence early in test, vanishing interference as $t \to \infty$.

### Verification & Unit Testing
* Added `TestSymmetricBMatrixAndReset` to [test_dynamic_learning_rate.py](file:///c:/Users/ravik/OneDrive/Desktop/college/ipd_curr/codes/test_dynamic_learning_rate.py) (12/12 unit tests passing).
* Multi-step diagnostic confirmed monotonic convergence (1.0803 $\to$ 0.9829) with $\mathbf{B}$ perfectly centered around zero (mean $= +0.0050$, $\min = -0.0505$, $\max = +0.0882$).

### Final 200-Question Paper Suite Benchmark ($N=200$, 200 Learners, Seed 42)

| System | RMSE | MAE | ECE | F1 | Macro AUROC | Trajectory Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **B4_MIRT** | 0.8811 | 0.7745 | 0.0486 | 0.2320 | 0.5000 | Strictly Monotonic |
| **P7_Full_Neuronotes** | **0.8515** | **0.7369** | 0.0537 | **0.2870** | **0.7176** | **Optimal Strictly Monotonic** |
| **C1_NoPrereq** | **0.8450** | **0.7301** | 0.0447 | **0.3897** | **0.7124** | **Optimal Strictly Monotonic** |
| **C2_NoCMatrix** | 0.8601 | 0.7483 | 0.0350 | 0.2320 | 0.5000 | Strictly Monotonic |
| **C3_FixedC** | **0.8496** | **0.7343** | 0.0505 | **0.3161** | **0.7099** | **Optimal Strictly Monotonic** |
| **C4_NoSMD** | 0.9014 | 0.7899 | 0.0427 | 0.2320 | 0.5000 | Strictly Monotonic |
| **C5_NoRouter** | **0.8541** | **0.7366** | 0.0521 | **0.3194** | **0.7101** | **Optimal Strictly Monotonic** |

#### Key Outcomes:
1. **P7 Outperforms Standard MIRT**: Delta RMSE is **-0.0296** (0.8515 vs 0.8811), and Delta AUROC is **+0.2176** (0.7176 vs 0.5000).
2. **Elimination of Gradient Drift**: All 7 models converge strictly monotonically throughout all 200 questions without a single reversal.
3. **Calibrated Q-Matrix Validated**: P7 achieved an all-time low RMSE of **0.8515** while maintaining high diagnostic discrimination (**AUROC = 0.7176**).



