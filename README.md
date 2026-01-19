# Investment-potential-tracking

### 1. Installation

1. Download or clone the project folder to your local machine.
2. In MATLAB, add the following directories to your path:
   - `release_ipt_latest/code` (contains the main `run_ipt.m` entry point)
   - `scripts` (contains analysis and plotting tools)

### 2. Usage

The typical workflow involves two main steps: running experiments and analyzing results.

#### Step 1: Run Experiments
Use `run_ipt.m` to execute the IPT algorithm variants on the datasets.

```matlab
% Example: Run standard IPT variants
run_ipt('run_tag', 'experiment_v1', ...
        'structures', {struct('name', 'v1_base', 'func', @ipt_run)}, ...
        'dataset_names', {'djia', 'msci', 'tse'});
```

This will create a timestamped results folder under `release_ipt_latest/results_struct_selection_by_baseline_wins/`.

#### Step 2: Analyze Results & Generate Tables
After the run completes, use `ipt_analyze_results.m`. This script automatically:
1.  Computes performance metrics (Sharpe, Calmar, etc.) against baselines.
2.  Generates `stat_*.csv` summary files.
3.  Produces publication-ready LaTeX tables via `ipt_paper_tables.m`.

```matlab
% Example: Analyze a specific results folder
results_dir = 'release_ipt_latest/results_struct_selection_by_baseline_wins/run_experiment_v1_.../v1_base';
ipt_analyze_results('results_dir', results_dir);
```

The output LaTeX files (e.g., `paper_tables_minimal.tex`) are stored in the `docs/paper_tables` directory.

### 3. Datasets

The datasets used by the project are stored in the `Data Set` directory.

| Dataset | Region | Time span             | Period | Number of stocks |
| ------- | ------ | --------------------- | ------ | ---------------- |
| DJIA    | US     | 2001.01.14-2003.01.14 | 507    | 30               |
| NDX     | US     | 2012.12.04-2022.11.30 | 2516   | 25               |
| NYSE(O) | US     | 1962.07.03-1984.12.31 | 5651   | 36               |
| NYSE(N) | US     | 1985.01.01-2010.06.30 | 6431   | 23               |
| TSE     | CA     | 1994.01.04-1998.12.31 | 1259   | 88               |
| MSCI    | US     | 2006.04.01-2010.03.31 | 1043   | 24               |
| MARPD   | US     | 2010.01.05-2023.12.29 | 3412   | 30               |
