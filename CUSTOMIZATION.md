# Advanced Customization Guide

This guide provides detailed instructions for adapting the Automated Total Opportunity Case Study Generator to work with your specific data formats and analytical needs.

## 1. Mapping Your Input Data Columns

The most common customization is adapting the script to read CSV files with different column names. Instead of renaming your source files, you can define your column names in the `column_mapping` object within your `config.json` file.

This object is divided into three sections: `investment_file`, `performance_file`, and `generic_trends_file`.

**Note on Date Columns:** Regardless of the original column names you specify for `date_col` in each section, the script will automatically standardize them into a single column named `Date` after loading the data. This ensures internal consistency during the analysis.

**Column Resolution:** Column names are matched exactly first, then ignoring case and surrounding whitespace, and finally against common names (only when a single candidate matches). If nothing resolves, the run stops with an error naming the missing column.

**Date Parsing & Cadence:** Each date column is parsed with one format for the whole column (trailing times such as `00:00:00` are stripped, and BI export footer rows are dropped). If parsing is ambiguous, set the format explicitly in `date_formats`:
```json
"date_formats": {
  "investment_file": "%Y-%m-%d",
  "performance_file": "%d/%m/%Y",
  "generic_trends_file": "%Y-%m-%d"
}
```
The engine then detects the data cadence (daily, weekly, monthly) and drops partial periods at the edges. The investment and performance files **must share the same cadence**, otherwise the run stops with an error.

### a. Investment Data

In the `investment_file` section of the `column_mapping`, you can specify the names for the date, channel, and investment columns.

*   `date_col`: The name of the column containing the date of the investment.
*   `channel_col`: The name of the column containing the marketing channel or product group.
*   `investment_col`: The name of the column containing the investment amount.

**Example:**
If your investment file has columns named `day`, `cost`, and `channel`, you would configure it like this:
```json
"column_mapping": {
  "investment_file": {
    "date_col": "day",
    "channel_col": "channel",
    "investment_col": "cost"
  },
  ...
}
```

**Note:** The script automatically strips leading and trailing whitespace from the channel names found in the `channel_col`. This ensures that "Channel A" and "Channel A " are treated as the same channel.

### b. Performance Data

In the `performance_file` section, you specify which column from your data contains the primary business metric (e.g., "Conversions", "Leads", "Sessions"). This involves two parameters in your `config.json` that work together:

*   **`kpi_col`** (inside `column_mapping` -> `performance_file`): This is the **most critical** setting. It must match the **exact** column header in your performance CSV file. This tells the script which data to load and analyze.

*   **`performance_kpi_column`** (at the top level of the config): This sets a "friendly name" for your KPI that will be used in the titles and descriptions of the final reports.

For consistency, it is best practice to set both parameters to the same value.

**Example:**
If your performance CSV has a column named `Conversions` that you want to analyze, your `config.json` should be set up like this for maximum clarity in the reports:

```json
"performance_kpi_column": "Conversions",
...
"column_mapping": {
  "performance_file": {
    "date_col": "report_date",
    "kpi_col": "Conversions"
  },
  ...
}
```

### c. Generic Trends Data (Optional)

In the `generic_trends_file` section, you can specify the names for the date and trends columns. This file is used to provide the model with context about general market trends (e.g., search volume, competitor activity) that might influence your business outcomes.

**This input is optional.** If you do not provide a path in the `generic_trends_file_path` setting in your `config.json` (or leave it blank in the UI), the analysis will automatically synthesize generic control variables based on historical volume to ensure the causal model does not degrade.

*   `date_col`: The name of the column containing the date.
*   `trends_col`: The name of the column that provides general market data.

**Example:**
If your trends file uses `data` for the date and `buscas` for the trend data:
```json
"column_mapping": {
  "generic_trends_file": {
    "date_col": "data",
    "trends_col": "buscas"
  }
}
```

## 2. Choosing Your Optimization Goal

The analysis can be tailored to optimize for two different primary business goals: maximizing revenue or maximizing conversions (KPIs). You can control this with the `optimization_target` parameter in your `config.json`.

-   `"optimization_target": "REVENUE"` (Default)
    -   The analysis will focus on metrics like Revenue and iROI (Incremental Return on Investment).
    -   This mode requires `"average_ticket"` to be set to a value greater than 0.

-   `"optimization_target": "CONVERSIONS"`
    -   Use this when you don't have a reliable average ticket or your goal is purely to maximize the number of conversions.
    -   The analysis will focus on metrics like CPA (Cost Per Acquisition) and iCPA (Incremental Cost Per Acquisition).
    -   The `"average_ticket"` parameter will be ignored.

-   `"kpi_is_monetary": true`
    -   Use this when your KPI is already a monetary value (e.g., revenue in R$). Revenue is taken directly from the KPI, so `average_ticket` and `conversion_rate_from_kpi_to_bo` are ignored.
    -   This forces revenue/ROAS mode: if combined with `"CONVERSIONS"`, the target is ignored and a warning is logged.

**Example:**
```json
{
  "advertiser_name": "Advertiser B",
  "optimization_target": "CONVERSIONS",
  "financial_targets": {
    "target_cpa": 25.0,
    "target_icpa": 35.0
  },
  "p_value_threshold": 0.1,
  ...
}
```

## 3. Controlling the Bound of Analysis

By default, the analysis will explore investment saturation up to 200% of your highest historical daily spend to find the mathematical curve shape.

To cap this exploration dynamically, you can add the optional `investment_limit_factor` to your `config.json`.

-   `investment_limit_factor`: A number that multiplies your maximum historical daily investment to set an upper bound for the curves.

**How it Works:**
-   If your highest daily spend was $10,000 and you set `"investment_limit_factor": 1.5`, the saturation engines will only compute the response curves up to $15,000 per day.
-   If you omit this parameter, it will default to `2.0` (200%).
-   In the Streamlit UI, this value is auto-suggested from your investment history.

For the global channel-mix optimization, two extra optional limits apply:

-   `per_channel_investment_limit_factor` (default `3.0`): caps each channel's recommended spend at this multiple of its historical maximum.
-   `max_channel_mix_share` (default `0.4`): the maximum share of the total budget any single channel can receive in the optimized mix.

**Note:** When the global model has no out-of-sample predictive power, the engine does not recommend reallocating budget between channels and keeps the historical mix.

**Example:**
```json
{
  "advertiser_name": "Advertiser A",
  "average_ticket": 1000,
  "investment_limit_factor": 1.5,
  "p_value_threshold": 0.1,
  ...
}
```

## 4. Defining Your Financial Constraints (Guardrails)

A key feature of this engine is the ability to prune the saturation curves dynamically based on your actual business economics.

This is controlled by the `financial_targets` block in your `config.json`. You can mix and match any of these four filters. The engine will halt generating profitable opportunity points globally on any channel as soon as they violate any of your defined guardrails.

*   **`target_cpa`**: Sets a hard limit on the *average* Cost Per Acquisition across the entire investment.
*   **`target_icpa`**: Sets a hard limit on the *incremental* Cost Per Acquisition. It stops the engine from considering points if the cost to acquire the *next* single user is too high, even if the overall CPA is fine.
*   **`target_roas`**: Sets a limit based on the total Return on Ad Spend (requires `average_ticket`).
*   **`target_iroas`**: Sets a limit based on the *incremental* ROAS. It stops highlighting opportunities when an additional R$ 1.00 invested no longer returns enough revenue to justify the spend.

**How it Works:**
- If you set `"target_icpa": 30.0`, the point plotting on the saturation curve will structurally stop or flag exactly at the point where acquiring one more conversion dynamically exceeds R$ 30.00. 
- If you leave this block empty or set the values to `0` or `999999`, the engine assumes "infinite profitability" and will rely solely on the underlying curve's shape to guide validation.

**Example:**
```json
{
  "advertiser_name": "Advertiser A",
  "average_ticket": 1000,
  "conversion_rate_from_kpi_to_bo": 0.015,
  "financial_targets": {
    "target_cpa": 15.0,
    "target_icpa": 25.0,
    "target_roas": 5.0,
    "target_iroas": 2.0
  },
  "p_value_threshold": 0.1,
  ...
}
```

## 5. Data Cleaning & Outlier Treatment

Real-world data often contains anomalies or "spikes" (e.g., a tracking bug causing 10x traffic for one day) that can distort statistical models. To robustly handle such data, the engine includes a configurable outlier treatment feature.

You can control this behavior using the `treat_outliers` parameter in your `config.json`.

*   **`"treat_outliers": true`**: (Boolean) Automatically detects and caps outliers in your primary KPI column (the one defined in `performance_kpi_column`) using the Interquartile Range (IQR) method (1.5 * IQR).
*   **`"treat_outliers": ["Sessions", "Conversions"]`**: (List of Strings) Allows you to specify exactly which columns in your performance data should be treated.
*   **`"treat_outliers": false`** (Default): No outlier treatment is applied. Use this if your data spikes are genuine (e.g., successful campaigns) and you want the Causal Impact model to fully account for them natively.

**Example:**
```json
{
  "advertiser_name": "Advertiser C",
  "treat_outliers": ["Sessions"],
  ...
}
```

## 6. Event Detection & Causal Validation

These parameters control which investment changes become events and which events are considered validated.

*   **`increase_threshold_percent`** / **`decrease_threshold_percent`**: The minimum investment change (in %) that flags an event. Very low values produce noisy events and log a warning.
*   **`post_event_days`**: The length of the post-event window analyzed.
*   **`min_pre_period_days`** (default `14`): The minimum history required before an event. It is converted into periods using the detected cadence, with a floor of 8 periods.
*   **`max_events_to_analyze`**: The maximum number of candidate events analyzed per run.

An event is only validated when it passes every enabled gate:

*   **`require_statistical_significance`** (default `true`): The p-value, derived from the model's own uncertainty, must be below `p_value_threshold`.
*   **`require_logical_direction`** (default `true`): The lift must move in the same direction as the investment change.
*   **`require_model_fit`** (default `true`): The model's R² must be at least `r_squared_threshold` (default `0.3`).

In the Streamlit UI, `min_pre_period_days`, the increase/decrease thresholds and `investment_limit_factor` are auto-suggested from your investment history.

**Example:**
```json
{
  "p_value_threshold": 0.1,
  "r_squared_threshold": 0.3,
  "min_pre_period_days": 14,
  "require_statistical_significance": true,
  "require_logical_direction": true,
  "require_model_fit": true,
  "increase_threshold_percent": 50,
  "decrease_threshold_percent": 30,
  "post_event_days": 14,
  "max_events_to_analyze": 5
}
```
