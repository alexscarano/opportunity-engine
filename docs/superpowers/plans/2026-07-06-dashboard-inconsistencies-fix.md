# Dashboard Elasticity Inconsistencies Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 5 confirmed bugs behind the inconsistent charts/tables in the Elasticidade dashboard tab: the Stage-2 Ridge model collapsing onto one channel, `MinMaxScaler` extrapolation producing negative projected KPIs, a broken saturation-point heuristic, a misleading `iCPA = R$0,00` fallback, and an inaccurate chart caption.

**Architecture:** Extract the risky/testable pieces of logic into small, pure, unit-testable functions (in `elasticity_analysis.py` for the modeling fixes, in `dashboard_charts.py` for the dashboard-data-derivation fixes — both files already import cleanly into their callers without circular deps), test each in isolation with synthetic data, then wire each into its single call site with a minimal diff. No new files, no new config keys, no changes to `analysis.py`/`saturation_curve.py`/`local_main*.py`.

**Tech Stack:** Python 3.13, pandas, numpy, scikit-learn (`Ridge`, `MinMaxScaler`), scipy (`minimize`), pytest via `unittest.TestCase` (matches existing test style in this repo).

**Reference:** Full investigation and rationale in `docs/superpowers/specs/2026-07-06-dashboard-inconsistencies-fix-design.md`.

---

## Before you start

Line numbers cited below (e.g. `scripts/elasticity_analysis.py:246-261`) reflect the file's state at
the time this plan was written. Earlier tasks in this same file (`elasticity_analysis.py` in Tasks
1-4, `streamlit_app.py` in Tasks 6/8/9) insert new lines, so later tasks' line numbers will have
shifted by however many lines were added. Use the "Current code" block shown in each step to locate
the exact spot to edit — the line number is a locator hint, the code block is the source of truth.

Run the existing suite once to confirm a clean baseline:

```bash
mise exec -- uv run --with pytest pytest tests/test_elasticity_analysis.py tests/test_dashboard_charts.py -v
```

Expected: `23 passed`.

---

### Task 1: `optimize_with_restarts` helper (fixes the Ridge collapse)

**Files:**
- Modify: `scripts/elasticity_analysis.py` (add new function near `elasticity_objective_function`, i.e. after line 152/before `run_mmm_engine`)
- Test: `tests/test_elasticity_analysis.py`

The Stage-2 optimization in `run_mmm_engine` fits 39 parameters (alpha/k/s × up to 13 channels) with a single `scipy.minimize()` call from one heuristic starting point. On the real dataset this collapses to 100% of the marketing credit on one channel and 0% on the rest (confirmed in the design spec). A multi-start search — the existing heuristic guess plus several randomized starts within the same bounds, keeping whichever converges to the lowest error — reduces the chance of landing in a degenerate local minimum.

- [ ] **Step 1: Write the failing test**

Current top of `tests/test_elasticity_analysis.py` (lines 1-10):

```python
# -*- coding: utf-8 -*-
import sys
import os
import unittest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from elasticity_analysis import cap_channel_mix_share
```

Replace with:

```python
# -*- coding: utf-8 -*-
import sys
import os
import unittest

import numpy as np
from scipy.optimize import minimize

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from elasticity_analysis import cap_channel_mix_share, optimize_with_restarts
```

Then append this to the end of the file (before the `if __name__ == "__main__":` block):

```python
def two_basin_objective(params, _):
    """Toy non-convex objective with a shallow local minimum near x=1 (value 0.5)
    and a deeper global minimum near x=8 (value 0), used to prove multi-start
    escapes a bad local minimum a single start would get stuck in."""
    x = params[0]
    return min((x - 1) ** 2 + 0.5, (x - 8) ** 2)


class TestOptimizeWithRestarts(unittest.TestCase):

    def test_finds_better_minimum_than_a_single_bad_start(self):
        bounds = [(0.0, 10.0)]
        bad_start = [1.0]

        single_start_result = minimize(
            two_basin_objective, bad_start, args=(None,), bounds=bounds, method="L-BFGS-B"
        )
        multi_start_result = optimize_with_restarts(
            two_basin_objective, bad_start, bounds, args=(None,), n_restarts=5, seed=42
        )

        self.assertLess(multi_start_result.fun, single_start_result.fun)
        self.assertAlmostEqual(multi_start_result.fun, 0.0, places=4)

    def test_reproducible_with_a_fixed_seed(self):
        bounds = [(0.0, 10.0)]
        result_a = optimize_with_restarts(
            two_basin_objective, [1.0], bounds, args=(None,), n_restarts=5, seed=42
        )
        result_b = optimize_with_restarts(
            two_basin_objective, [1.0], bounds, args=(None,), n_restarts=5, seed=42
        )
        self.assertEqual(result_a.fun, result_b.fun)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_elasticity_analysis.py -v`
Expected: FAIL with `ImportError: cannot import name 'optimize_with_restarts'`

- [ ] **Step 3: Write the implementation**

In `scripts/elasticity_analysis.py`, add this function after `elasticity_objective_function` (after line 152, before `def run_mmm_engine(config):`):

```python
def optimize_with_restarts(objective, initial_params, bounds, args, n_restarts=5, seed=42):
    """
    Runs scipy.optimize.minimize from `initial_params` plus `n_restarts` additional
    starting points sampled uniformly within `bounds`, and returns whichever run
    converged to the lowest objective value.

    Guards against the optimizer settling into a bad local minimum from a single
    starting point -- which is what happens with the Stage-2 elasticity fit on
    collinear channel spend (see docs/superpowers/specs/2026-07-06-dashboard-inconsistencies-fix-design.md).
    """
    rng = np.random.default_rng(seed)
    candidate_starts = [initial_params] + [
        [rng.uniform(low, high) for low, high in bounds] for _ in range(n_restarts)
    ]

    best_result = None
    for candidate in candidate_starts:
        candidate_result = minimize(
            objective,
            candidate,
            args=args,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 500, "disp": False},
        )
        if best_result is None or candidate_result.fun < best_result.fun:
            best_result = candidate_result

    return best_result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_elasticity_analysis.py -v`
Expected: `PASSED` for both new tests, all previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/elasticity_analysis.py tests/test_elasticity_analysis.py
git commit -m "feat(elasticity): add multi-start optimization helper"
```

---

### Task 2: Wire multi-start into `run_mmm_engine`

**Files:**
- Modify: `scripts/elasticity_analysis.py:246-261`

- [ ] **Step 1: Replace the single-start optimization call**

Current code (`scripts/elasticity_analysis.py:246-261`):

```python
    # Run optimization
    result = minimize(
        elasticity_objective_function,
        initial_params,
        args=(df, y_lift, active_spend_cols),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 500, "disp": False},
    )

    if not result.success:
        print(
            f"   - WARNING: Optimization did not converge fully. Details: {result.message}"
        )

    optimal_params = result.x
```

Replace with:

```python
    # Run optimization from multiple starting points to avoid a bad local minimum
    result = optimize_with_restarts(
        elasticity_objective_function,
        initial_params,
        bounds,
        args=(df, y_lift, active_spend_cols),
    )

    if not result.success:
        print(
            f"   - WARNING: Optimization did not converge fully. Details: {result.message}"
        )

    optimal_params = result.x
```

- [ ] **Step 2: Run the full test suite to confirm nothing broke**

Run: `mise exec -- uv run --with pytest pytest tests/test_elasticity_analysis.py tests/test_dashboard_charts.py -v`
Expected: all tests pass (this function has no direct unit test — it's exercised end-to-end by the CLI pipeline, which needs real CSV input, out of scope for this plan's test suite).

- [ ] **Step 3: Commit**

```bash
git add scripts/elasticity_analysis.py
git commit -m "fix(elasticity): use multi-start optimization in Stage 2 fit"
```

---

### Task 3: `predict_clipped_kpi` helper (fixes negative-KPI extrapolation)

**Files:**
- Modify: `scripts/elasticity_analysis.py` (add new function near `hill_transform`)
- Test: `tests/test_elasticity_analysis.py`

`simulate_kpi` and `generate_individual_response_curves` call `mkt_scaler.transform()` with simulated spend that can fall outside the range the scaler was fit on (e.g. spend=0, when the channel's historical minimum was always > 0). `MinMaxScaler` does not clip, so this can produce a negative scaled feature; multiplied by the model's non-negative coefficient, that drags the predicted KPI below the organic baseline — even negative, which was confirmed in `outputs/user_1/Meu_Projeto_Dynamic_2_dynamic/global_saturation_analysis/response_curve_data.csv` (`Projected_Total_KPIs = -141394.67` at zero investment). Clipping the scaled features to `[0, 1]` bounds the simulation to the physically sensible range.

- [ ] **Step 1: Write the failing test**

In `tests/test_elasticity_analysis.py`, change the import line added in Task 1:

```python
from elasticity_analysis import cap_channel_mix_share, optimize_with_restarts
```

to:

```python
from elasticity_analysis import cap_channel_mix_share, optimize_with_restarts, predict_clipped_kpi
```

Then append this to the end of the file (before the `if __name__ == "__main__":` block):

```python
class FakeScaler:
    """Replicates MinMaxScaler's un-clipped affine transform for a column whose
    historical range was [100, 200], without depending on sklearn internals."""

    def transform(self, X):
        return np.array([[(row[0] - 100.0) / (200.0 - 100.0)] for row in X])


class FakeModel:
    """Replicates a fitted single-feature linear model with coefficient 10."""

    def predict(self, X):
        return np.array([10.0 * row[0] for row in X])


class TestPredictClippedKpi(unittest.TestCase):

    def test_clips_extrapolated_feature_instead_of_going_negative(self):
        # Simulated spend of 0 is below the historical minimum (100), which would
        # scale to -1.0 without clipping -> KPI 10 below baseline. Clipped to 0,
        # it should contribute nothing, landing exactly on the baseline.
        predicted = predict_clipped_kpi(500.0, FakeModel(), FakeScaler(), [0.0])
        self.assertEqual(predicted, 500.0)

    def test_leaves_in_range_values_unaffected(self):
        # Spend of 150 scales to 0.5, well within [0, 1] -- clip is a no-op.
        predicted = predict_clipped_kpi(500.0, FakeModel(), FakeScaler(), [150.0])
        self.assertEqual(predicted, 505.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_elasticity_analysis.py -v`
Expected: FAIL with `ImportError: cannot import name 'predict_clipped_kpi'`

- [ ] **Step 3: Write the implementation**

In `scripts/elasticity_analysis.py`, add this function after `hill_transform` (after line 120, before `elasticity_objective_function`):

```python
def predict_clipped_kpi(organic_baseline_mean, mkt_model, mkt_scaler, mkt_features):
    """
    Predicts KPI from Hill-transformed marketing features, clipping the scaled
    features to [0, 1] before feeding the linear model.

    Simulated spend outside the historically observed range (e.g. investment=0,
    or several multiples of the historical average) extrapolates `mkt_scaler`
    beyond the range it was fit on -- MinMaxScaler does not clip, so this can
    produce a negative scaled feature which, combined with the model's
    non-negative coefficients, predicts a KPI below the organic baseline
    (even negative). Clipping keeps the simulation within the physically
    sensible 0%-100%-of-historical-saturation range.
    """
    scaled_features = np.clip(mkt_scaler.transform([mkt_features]), 0.0, 1.0)
    return organic_baseline_mean + mkt_model.predict(scaled_features)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_elasticity_analysis.py -v`
Expected: `PASSED` for both new tests, all previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/elasticity_analysis.py tests/test_elasticity_analysis.py
git commit -m "feat(elasticity): add clipped KPI prediction helper"
```

---

### Task 4: Wire `predict_clipped_kpi` into the two simulation call sites

**Files:**
- Modify: `scripts/elasticity_analysis.py:452-465` (`simulate_kpi`, inside `generate_aggregated_response_curve`)
- Modify: `scripts/elasticity_analysis.py:749-752` (inside `generate_individual_response_curves`)

- [ ] **Step 1: Update `simulate_kpi`**

Current code (`scripts/elasticity_analysis.py:452-465`):

```python
    def simulate_kpi(total_spend, mix):
        mkt_features = []
        for i, col in enumerate(active_spend_cols):
            simulated_daily_spend = total_spend * mix.get(col, 0)
            simulated_adstocked = simulated_daily_spend * adstock_multipliers[col]
            mkt_features.append(
                hill_transform(
                    simulated_adstocked, opt_params["ks"][i], opt_params["ss"][i]
                )
            )
        return (
            organic_baseline_mean
            + mkt_model.predict(mkt_scaler.transform([mkt_features]))[0]
        )
```

Replace with:

```python
    def simulate_kpi(total_spend, mix):
        mkt_features = []
        for i, col in enumerate(active_spend_cols):
            simulated_daily_spend = total_spend * mix.get(col, 0)
            simulated_adstocked = simulated_daily_spend * adstock_multipliers[col]
            mkt_features.append(
                hill_transform(
                    simulated_adstocked, opt_params["ks"][i], opt_params["ss"][i]
                )
            )
        return predict_clipped_kpi(organic_baseline_mean, mkt_model, mkt_scaler, mkt_features)
```

- [ ] **Step 2: Update the loop in `generate_individual_response_curves`**

Current code (`scripts/elasticity_analysis.py:749-752`):

```python
            predicted_kpi = (
                organic_baseline_mean
                + mkt_model.predict(mkt_scaler.transform([mkt_features]))[0]
            )
```

Replace with:

```python
            predicted_kpi = predict_clipped_kpi(
                organic_baseline_mean, mkt_model, mkt_scaler, mkt_features
            )
```

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

Run: `mise exec -- uv run --with pytest pytest tests/test_elasticity_analysis.py tests/test_dashboard_charts.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/elasticity_analysis.py
git commit -m "fix(elasticity): clip simulated KPI predictions to avoid extrapolation artifacts"
```

---

### Task 5: `find_saturation_point` helper (fixes the broken saturation heuristic)

**Files:**
- Modify: `scripts/dashboard_charts.py` (add new function; it doesn't build a `Figure`, but this is the only module already imported by `streamlit_app.py` for pure, tested data-derivation logic)
- Test: `tests/test_dashboard_charts.py`

The current heuristic (in `streamlit_app.py`) compares every step's marginal gain against the derivative of the **first** step, and stops at the first crossing. Two problems: (1) that first-step reference is exactly where the extrapolation bug from Task 3/4 produced a huge spurious spike, and (2) even without that spike, "stop at the first dip below threshold" can trip on a transient dip that recovers afterward, landing on an isolated blip instead of the point where the curve is genuinely, sustainedly flat. The fix: use the curve's peak marginal gain as the 100% reference (robust to whichever index the peak happens to sit at), and find the *last* index still above threshold — saturation begins right after that, since everything beyond it stays flat.

- [ ] **Step 1: Write the failing test**

Add to the import block at the top of `tests/test_dashboard_charts.py` (currently lines 11-22):

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
    find_saturation_point,
)
```

Append this test class:

```python
class TestFindSaturationPoint(unittest.TestCase):

    def test_skips_transient_dip_and_finds_start_of_sustained_flat_region(self):
        # Marginal gain (per R$10 of investment) is 100, 90, then dips to 5 at
        # investment=30 (a transient blip) before recovering to 80, 70, 60, and
        # only truly flattens (3, 2, 1) from investment=70 onward.
        df = pd.DataFrame({
            "Daily_Investment": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "Projected_Total_KPIs": [0, 1000, 1900, 1950, 2750, 3450, 4050, 4080, 4100, 4110],
        })
        optimal_point = df.iloc[-1]

        result = find_saturation_point(df, optimal_point)

        # A naive "first dip below threshold" scan would wrongly stop at 30;
        # the real, sustained flattening only starts at 70.
        self.assertEqual(result["Daily_Investment"], 70)

    def test_falls_back_to_optimal_point_when_curve_never_rises(self):
        df = pd.DataFrame({
            "Daily_Investment": [0, 10, 20],
            "Projected_Total_KPIs": [100, 90, 80],
        })
        optimal_point = df.iloc[-1]

        result = find_saturation_point(df, optimal_point)

        self.assertEqual(result["Daily_Investment"], optimal_point["Daily_Investment"])

    def test_falls_back_to_optimal_point_when_curve_is_too_short(self):
        df = pd.DataFrame({
            "Daily_Investment": [0],
            "Projected_Total_KPIs": [100],
        })
        optimal_point = df.iloc[-1]

        result = find_saturation_point(df, optimal_point)

        self.assertEqual(result["Daily_Investment"], optimal_point["Daily_Investment"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: FAIL with `ImportError: cannot import name 'find_saturation_point'`

- [ ] **Step 3: Write the implementation**

Add to `scripts/dashboard_charts.py`, after the imports (before `build_icpa_curve`):

```python
def find_saturation_point(df, optimal_point):
    """
    Finds the point on the aggregate response curve (sorted by ascending
    investment) where marginal KPI gain has sustainedly collapsed to under 10%
    of the curve's peak marginal gain.

    Uses the curve's own peak as the 100% reference (not the first step, which
    can be an extrapolation artifact -- see predict_clipped_kpi in
    elasticity_analysis.py) and returns the point right after the *last* index
    still above threshold, so a transient dip that recovers afterward doesn't
    get mistaken for the real, sustained saturation ceiling.

    Falls back to `optimal_point` when the curve never rises (or is too short
    to compute a derivative from).
    """
    incremental_kpis = df["Projected_Total_KPIs"].diff().fillna(0).values
    investment_steps = df["Daily_Investment"].diff().fillna(1).values
    first_derivative = incremental_kpis / investment_steps

    if len(first_derivative) <= 1:
        return optimal_point

    peak_marginal_gain = first_derivative.max()
    if peak_marginal_gain <= 0:
        return optimal_point

    saturation_threshold = peak_marginal_gain * 0.1
    above_threshold = np.where(first_derivative > saturation_threshold)[0]
    if len(above_threshold) == 0:
        return optimal_point

    saturation_idx = above_threshold[-1] + 1
    if saturation_idx >= len(df):
        return optimal_point

    return df.iloc[saturation_idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: `PASSED` for all three new tests, all previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add robust saturation-point detection helper"
```

---

### Task 6: Wire `find_saturation_point` into `streamlit_app.py`

**Files:**
- Modify: `scripts/streamlit_app.py:267-278` (import block)
- Modify: `scripts/streamlit_app.py:1421-1437` (saturation-point computation)

- [ ] **Step 1: Add the import**

Current code (`scripts/streamlit_app.py:267-278`):

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
)
```

Replace with:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
    find_saturation_point,
)
```

- [ ] **Step 2: Replace the inline saturation-point computation**

Current code (`scripts/streamlit_app.py:1421-1437`):

```python
                optimal_point = filtered_df.iloc[-1]

                incremental_kpis = df["Projected_Total_KPIs"].diff().fillna(0).values
                investment_steps = df["Daily_Investment"].diff().fillna(1).values
                first_derivative = incremental_kpis / investment_steps
                saturation_point = optimal_point
                if len(first_derivative) > 1:
                    initial_marginal_gain = first_derivative[1]
                    if initial_marginal_gain > 0:
                        saturation_threshold = initial_marginal_gain * 0.1
                        sat_indices = np.where(
                            first_derivative[1:] < saturation_threshold
                        )[0]
                        if len(sat_indices) > 0:
                            saturation_idx = sat_indices[0] + 1
                            if saturation_idx < len(df):
                                saturation_point = df.iloc[saturation_idx]
```

Replace with:

```python
                optimal_point = filtered_df.iloc[-1]

                saturation_point = find_saturation_point(df, optimal_point)
```

- [ ] **Step 3: Sanity-check the module still imports cleanly**

Run: `mise exec -- uv run python -c "import sys; sys.path.append('scripts'); import ast; ast.parse(open('scripts/streamlit_app.py', encoding='utf-8').read())"`
Expected: no output (parses without a `SyntaxError`). `streamlit_app.py` itself isn't unit-tested (no `tests/test_streamlit_app.py`), so a full run requires `mise run dev` and manual verification in the browser (covered by the Verification section at the end of this plan).

- [ ] **Step 4: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "fix(dashboard): use robust saturation-point detection in Elasticidade tab"
```

---

### Task 7: `compute_incremental_cpa` helper (fixes the misleading `iCPA = R$0,00`)

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Test: `tests/test_dashboard_charts.py`

The scenario table computes `iCPA = np.where(kpi_incremental > 0, investimento_incremental / kpi_incremental, 0)`. When a scenario's incremental KPI isn't positive (as currently happens for the broken "Cenário de Saturação" — see Task 5/6), this forces a literal `R$ 0,00` instead of leaving the ratio undefined, which reads as "free extra KPIs" instead of "not a meaningful ratio." `streamlit_app.py`'s own `format_currency` (`scripts/streamlit_app.py:1489-1498`) already renders `NaN` as `"N/A"` — no change needed there, only the value fed into it.

- [ ] **Step 1: Write the failing test**

Add `compute_incremental_cpa` to the import block added in Task 5 (`tests/test_dashboard_charts.py`):

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
    find_saturation_point,
    compute_incremental_cpa,
)
```

Append this test class:

```python
class TestComputeIncrementalCpa(unittest.TestCase):

    def test_returns_ratio_when_kpi_incremental_is_positive(self):
        result = compute_incremental_cpa(
            pd.Series([1000.0, 1500000.0]), pd.Series([100.0, 142600.0])
        )
        self.assertAlmostEqual(result[0], 10.0)
        self.assertAlmostEqual(result[1], 1500000.0 / 142600.0)

    def test_returns_nan_when_kpi_incremental_is_not_positive(self):
        # Mirrors the broken "Cenário de Saturação" row: both incremental
        # investment and incremental KPI negative -- not a meaningful ratio,
        # must not silently become a literal 0.
        result = compute_incremental_cpa(
            pd.Series([-2984379.93, 500.0]), pd.Series([-850025.0, 0.0])
        )
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_incremental_cpa'`

- [ ] **Step 3: Write the implementation**

Add to `scripts/dashboard_charts.py`, next to `find_saturation_point`:

```python
def compute_incremental_cpa(investimento_incremental, kpi_incremental):
    """
    Ratio of incremental investment to incremental KPI ("iCPA"), or NaN when
    the scenario didn't produce a positive KPI gain.

    A non-positive incremental KPI makes the ratio not meaningful (it is not
    a real, free-or-cheap cost -- it's a scenario that performs worse than the
    baseline), so it must not be silently rendered as a literal 0.
    """
    return np.where(kpi_incremental > 0, investimento_incremental / kpi_incremental, np.nan)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: `PASSED` for both new tests, all previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add incremental CPA helper that distinguishes N/A from 0"
```

---

### Task 8: Wire `compute_incremental_cpa` into the scenario table

**Files:**
- Modify: `scripts/streamlit_app.py` (import block added in Task 6)
- Modify: `scripts/streamlit_app.py:1478-1483`

- [ ] **Step 1: Add the import**

Current code (added in Task 6):

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
    find_saturation_point,
)
```

Replace with:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
    find_saturation_point,
    compute_incremental_cpa,
)
```

- [ ] **Step 2: Replace the inline iCPA computation**

Current code (`scripts/streamlit_app.py:1478-1483`):

```python
                scenario_df["iCPA"] = np.where(
                    scenario_df[f"{kpi_name} Incrementais"] > 0,
                    scenario_df["Investimento Incremental"]
                    / scenario_df[f"{kpi_name} Incrementais"],
                    0,
                )
```

Replace with:

```python
                scenario_df["iCPA"] = compute_incremental_cpa(
                    scenario_df["Investimento Incremental"],
                    scenario_df[f"{kpi_name} Incrementais"],
                )
```

- [ ] **Step 3: Confirm the baseline row is still explicitly zeroed**

Directly below (`scripts/streamlit_app.py:1485-1487`), this must remain unchanged — it is correct as-is (the baseline scenario's incrementals over itself are exactly zero, not NaN):

```python
                scenario_df.loc[
                    0, ["Investimento Incremental", f"{kpi_name} Incrementais", "iCPA"]
                ] = 0.0
```

- [ ] **Step 4: Sanity-check the module still parses**

Run: `mise exec -- uv run python -c "import ast; ast.parse(open('scripts/streamlit_app.py', encoding='utf-8').read())"`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "fix(dashboard): show N/A instead of R\$ 0,00 for non-positive iCPA scenarios"
```

---

### Task 9: Fix the misleading "Evolução do Mix de Canais" caption

**Files:**
- Modify: `scripts/streamlit_app.py:1746-1751`

The chart's supporting text claims the recommended allocation "muda à medida que o orçamento total escala" (changes as the budget scales). But `build_channel_mix_evolution` (`scripts/dashboard_charts.py:130-200`) plots `Spend_{channel}_Strategic = Daily_Investment × mix_fixo` for a fixed `strategic_mix` — a constant proportion multiplied by a scaling total always yields the same percentage at every point. The chart cannot show what the caption promises; this is a trivial text fix, no logic changes, no test needed.

- [ ] **Step 1: Replace the caption**

Current code (`scripts/streamlit_app.py:1746-1751`):

```python
                st.markdown("### Evolução do Mix de Canais por Orçamento")
                st.markdown(
                    "Mostra como a alocação recomendada entre canais (Modelo de Elasticidade) muda "
                    "à medida que o orçamento total escala — útil para saber quais canais absorvem "
                    "mais verba incremental conforme você investe mais."
                )
```

Replace with:

```python
                st.markdown("### Evolução do Mix de Canais por Orçamento")
                st.markdown(
                    "Mostra a alocação recomendada entre canais (Modelo de Elasticidade) — uma "
                    "proporção fixa — em reais por canal, conforme o orçamento total escala. As "
                    "porcentagens permanecem constantes; o que muda é o valor em R$ que cada canal "
                    "recebe a cada nível de investimento total."
                )
```

- [ ] **Step 2: Sanity-check the module still parses**

Run: `mise exec -- uv run python -c "import ast; ast.parse(open('scripts/streamlit_app.py', encoding='utf-8').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "fix(dashboard): correct misleading channel-mix-evolution caption"
```

---

## Final Verification

- [ ] **Step 1: Run the full test suite**

Run: `mise exec -- uv run --with pytest pytest tests/ -v`
Expected: all tests pass (23 pre-existing + 9 new = 32).

- [ ] **Step 2: Manually verify in the running dashboard**

Run: `mise run dev` (starts `streamlit run scripts/streamlit_app.py`), open the app, load one of the existing advertiser outputs (e.g. `Meu_Projeto_Dynamic_2_dynamic`), go to the Elasticidade tab, and check:
- "Resumo dos Cenários Projetados" table: "Cenário de Saturação" now shows an investment level *above* "Cenário Atual" (not below it), and its iCPA is either a real number or "N/A" — never a suspicious "R$ 0.00" next to negative incrementals.
- "Comparativo de Saturação entre Canais": more than one channel should show visible movement (not just Skyscanner flat-lining everyone else at 0%).
- "Curva de Resposta: <canal>" dropdown: pick a previously-flat channel (e.g. AWIN, BING) and confirm the curve now shows some response shape instead of a perfectly flat line.
- "Evolução do Mix de Canais por Orçamento": caption now matches what the chart actually shows.

This step can't be scripted — note any remaining visual oddities for a follow-up investigation rather than treating them as this plan's regression.

Note: because Task 1/2's multi-start fix only takes effect the next time the CLI pipeline (`local_main.py` / `local_main-without-gemini.py`) regenerates `outputs/<advertiser>/global_saturation_analysis/*.csv`, the dashboard needs those outputs regenerated to reflect the fix — re-running against the existing `inputs/` configs is required before this manual check will show the corrected model output.
