# PyCAALP Experiments

This directory contains experimental validation of the assembly line scheduling algorithms, with emphasis on the **strategy comparison**, **bl_union convergence**, and **adaptive convergence** studies that form the core of the JMS publication.

## Overview

The experiments evaluate different solution strategies along three key dimensions:

1. **Computational efficiency** — how the subgraph-MIP strategies scale vs. solving the full MIP
2. **Solution quality** — objective gap to optimality and quality-cost tradeoffs
3. **Convergence behavior** — how solution quality improves as the subgraph expands (k-shortest paths)

---

## Core Experiments

### 1. Strategy Comparison

**Directory:** [`strategy_comparison/`](strategy_comparison/)

**What it tests:** Compares six solution strategies on assembly_1 with varying k (number of shortest paths) and λ (time-balance parameter).

**The strategies:**

1. **Full MIP** — Joint path-selection + phase-assignment on the complete digraph (oracle baseline)
2. **Path-Enum (edge_weight)** — k-shortest paths enumerated with engineering weight
3. **Path-Enum (time_balanced_weight)** — k-shortest paths enumerated with time-balanced weight
4. **Subgraph MIP (edge_weight)** — MIP on union of k paths with engineering weight
5. **Subgraph MIP (time_balanced_weight)** — MIP on union of k paths with time-balanced weight
6. **Subgraph MIP (blended-union)** — MIP on blended union of k paths (the recommended strategy)

**Run:**

```bash
python -m experiments.strategy_comparison.test_strategy_comparison
```

**Output:** `strategy_comparison/strategy_comparison.csv`  
Contains: wall-clock time, objective value, subgraph size (% of edges), and gap to full MIP for each (k, λ) pair and strategy.

**Key findings:** The blended-union strategy achieves <1% gap to the full MIP while solving 2–3× faster on medium-sized instances.

---

### 2. Blended-Union Convergence

**Directory:** [`bl_union_convergence/`](bl_union_convergence/)

**What it tests:** How the objective value and subgraph size of the **bl-union strategy** converge as k grows. Answers: "What is the minimum k (or % of edges) needed for tight convergence?"

**Why this experiment:** The full MIP is an expensive oracle on large assemblies. This isolates the bl-union method and sweeps k on a geometric grid to isolate where the quality gain saturates.

**Convergence criteria (auto-stop):**

- **Objective plateau:** Relative improvement < 0.5% for 2 consecutive k values  
- **Edge saturation:** Subgraph gains no new edges (the union stopped growing)  
- **Safety cap:** Stops at k=5000 or when either criterion fires first

**Run (single λ):**

```bash
python -m experiments.bl_union_convergence.bl_union_convergence \
    --w-balanced 0.5 --num-phases 3 --out run/conv_0.5.csv
```

**Sweep across λ grid (parallel):**

```bash
cd experiments/bl_union_convergence && ./run_sweep.sh
```

**Output:** CSV with columns for k, subgraph_pct (% of edges), objective, gap to full MIP, solve time.

**Key finding:** On assembly_1, convergence happens early—often before 5% of edges are included. This enables the method to scale to larger problems.

---

### 3. Adaptive Convergence

**Directory:** [`adaptive_convergence/`](adaptive_convergence/)

**What it tests:** Convergence of the **adaptive subgraph strategy** on problems where the full MIP is unsolvable (high λ, large assemblies).

**Key difference from bl_union_convergence:** No oracle (full MIP). Instead, uses two **oracle-free bounds**:

- **Ideal objective:** Analytical lower bound computable in closed form  
- **Phase width:** Perfect-balance floor T_total/P for the makespan

**Metrics (both always ≥ 0):**

- **obj_vs_ideal_pct** — How far the subgraph objective sits above the analytical floor (method suboptimality)  
- **alpha_vs_width_pct** — How far the makespan sits above perfect balance

**Run (single λ):**

```bash
python -m experiments.adaptive_convergence.adaptive_convergence \
    --w-balanced 0.9 --num-phases 3 \
    --out experiments/adaptive_convergence/adaptive_convergence.csv
```

**Sweep across configurations:**

```bash
cd experiments/adaptive_convergence && ./run_sweep.sh
```

**Output:** Per-(assembly, phase count, λ) pair: timing breakdown, bound gaps, subgraph size at 10% edge limit.

**Key finding:** The adaptive method scales to high-λ regimes (λ → 1) where standard MIP solvers time out, with quality degradation quantified by the oracle-free bounds.

---

## Supporting Experiments

### Edge Reduction

**Directory:** [`edge_reduction/`](edge_reduction/)

Tests the effect of preprocessing edge reduction on solution quality and solve time. See [`edge_reduction/README.md`](edge_reduction/README.md).

### Multiple Attributes

**Directory:** [`multiple_attributes/`](multiple_attributes/)

Tests sensitivity to weights (w_tech, w_hand, w_tol, w_mass) on attribute changes. See [`multiple_attributes/README.md`](multiple_attributes/README.md).

### Multiple Stations

**Directory:** [`multiple_stations/`](multiple_stations/)

Tests scalability across problem instances with varying numbers of processing stations.

---

## Results Structure

Each experiment creates:

- **CSV results** — Main metrics table (parseable, version-controlled)
- **SVG/PNG plots** — Publication-ready visualizations  
- **Per-config folders** — Detailed logs and solve artifacts (timing, solver output, etc.)

Example for strategy_comparison:

```
strategy_comparison/
  ├── strategy_comparison.csv              # Main results
  ├── assembly_1_np_2/                     # Per-config dir
  │   ├── lambda_k_heatmap_*.svg          # Quality/time heatmaps
  │   ├── lambda_sweep_*.svg               # Sweep plots
  │   └── ...
  ├── assembly_1_np_3/
  └── ...
```

---

## Reproducing the Results

### Prerequisites

- Python 3.8+ with PyCAALP installed (see project root README)
- Gurobi solver (for the MIP subgraph-MIP strategy; free academic license available)
- ~30 GB disk for full results (plots + solver artifacts)

### Full Pipeline

1. **Strategy Comparison** (establishes that bl-union is the recommended method):

   ```bash
   python -m experiments.strategy_comparison.test_strategy_comparison
   ```

2. **BL-Union Convergence** (shows bl-union reaches tight quality early):

   ```bash
   cd experiments/bl_union_convergence && ./run_sweep.sh
   ```

3. **Adaptive Convergence** (shows method scales to unsolvable regions):

   ```bash
   cd experiments/adaptive_convergence && ./run_sweep.sh
   ```

4. **Plot generation** (if results already exist):

   ```bash
   cd experiments/[experiment_name] && python plot_convergence.py
   ```

---

## File Reference

| File | Purpose |
|------|---------|
| `strategy_comparison/test_strategy_comparison.py` | Main strategy comparison script |
| `strategy_comparison/plot_lambda_sweep.py` | Generates heatmaps and sweep plots |
| `bl_union_convergence/bl_union_convergence.py` | Convergence of bl-union method |
| `bl_union_convergence/plot_convergence.py` | Convergence curve plots |
| `adaptive_convergence/adaptive_convergence.py` | Oracle-free convergence study |
| `adaptive_convergence/plot_convergence.py` | Oracle-free bound plots |

---

## Publication Figures

The following results feed directly into the JMS submission:

1. **Strategy comparison** — Objective gap vs. k for all six strategies (assembly_1, multiple λ values)
2. **BL-Union convergence** — Subgraph size (% edges) and objective vs. k; stops when saturated
3. **Adaptive convergence** — Oracle-free bounds (obj_vs_ideal, alpha_vs_width) vs. k; shows scalability to high-λ regimes

---

## Questions?

- For methodology questions, see the docstrings in `bl_union_convergence.py` and `adaptive_convergence.py`
- For plotting, check the `plot_*.py` files in each experiment directory
- For configuration, edit the experiment scripts or pass CLI arguments (see `--help`)
