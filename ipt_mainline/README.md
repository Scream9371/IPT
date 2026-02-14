# IPT Mainline Reproduction (Simplified)

This folder is a migrated and simplified IPT mainline runner from `../IPT`.

Included:
- Mainline rolling 5y evaluation (4y train + 1y test)
- IPT metric aggregation (`CW`, `AnnRet`, `Sharpe`, `Sortino`, `Calmar`, `Sterling`, `MDD`)

Excluded:
- Baseline re-runs and baseline result generation
- Ablation experiments
- Statistical tests
- Sensitivity analysis

## Fixed mainline configuration

- `datasets = {'nyse-n','nyse-o','ndx','inv500','inv30','multi_asset'}`
- `q = 0.3`
- `risk_factor = 10`
- `reverse_factor = 10`
- `d_l = 252`
- `d_n = 21`
- `risk_gate = econ`
- `hold = 3`
- `L_pct = 99`
- `tran_cost = 0.001`
- rolling protocol: `4y train + 1y test`, step `1y`

## One-click run

From `Investment-potential-tracking` in MATLAB:

```matlab
out_dir = run();
```

Optional custom output directory:

```matlab
out_dir = run('out_dir', fullfile(pwd,'results','ipt_mainline_custom'));
```

Main outputs in `out_dir`:
- `run_config.json`
- `orig_rppt_rolling5y_summary.csv`
- `stat_cumwealth_rolling5y.csv`
- `stat_ann_ret_rolling5y.csv`
- `stat_sharpe_rolling5y.csv`
- `stat_sortino_rolling5y.csv`
- `stat_calmar_rolling5y.csv`
- `stat_sterling_rolling5y.csv`
- `stat_mdd_rolling5y.csv`
- `ipt-<dataset>_roll5y.mat`
