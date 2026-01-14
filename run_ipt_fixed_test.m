function run_ipt_fixed_test(varargin)
% run_ipt_fixed_test - IPT walk-forward selection of IPT-specific params with fixed win_size/epsilon.
%
% Fixed parameters (defaults):
%   win_size = 5
%   epsilon  = 100
%   tran_cost = 0.001
%
% Split rule (only dev/test):
%   split_mode='dev_test' with dev_ratio (default 0.6). The dev segment is used for warm-up + tuning
%   (warm-up portion is not scored), then the selected params are frozen on the test segment.
%
% Validation objective:
%   'log_wealth' (default): maximize mean(log(fold_terminal_wealth))
%   'calmar': maximize mean(log(fold_terminal_wealth)/max_drawdown)
%   'log_wealth_turnover': maximize mean(log(fold_terminal_wealth)) - lambda * mean(turnover_mean)
%       where turnover_mean is average sum(abs(b_t - b_prev)) over the scored days
%       (lambda is controlled by turnover_penalty_lambda).
%   'log_wealth_q25': maximize the 25th percentile of log(fold_terminal_wealth) (robust to unstable folds)
%   'log_wealth_min': maximize min(log(fold_terminal_wealth)) (maximin; strongly penalizes collapses)
%   'log_wealth_last': maximize log(terminal_wealth) on the LAST validation fold (recent-fold tuning for non-stationary markets)
%   'log_wealth_recent': maximize weighted mean(log(fold_terminal_wealth)) with weights increasing over folds (recent folds emphasized)
%   'log_wealth_stable': maximize mean(log(fold_terminal_wealth)) - 0.5*std(log(fold_terminal_wealth)) (stability-penalized)
%   'log_wealth_plus_sharpe': maximize mean(log(fold_terminal_wealth)) + w * mean(fold_sharpe), where w is val_sharpe_weight
%   'rank_wealth_sharpe': maximize -(rank(log_wealth)+rank(sharpe)) on the validation folds,
%       where ranks are computed across grid candidates (lower rank is better).
%   'log_then_sharpe': two-stage selection: topN by log_wealth, then best by sharpe (tie-break by log_wealth).
%   'log_then_wins_both': two-stage selection: topN by log_wealth, then best by wins_both (tie-break by log_wealth),
%       where wins_both is the number of OLPS baselines beaten on BOTH metrics (log_wealth and sharpe) on validation folds.
%   'wins_both': directly maximize wins_both on validation folds (tie-break by log_wealth, then sharpe).
%
% Grid profile:
%   'robust' (default): smaller grid to reduce overfitting risk and runtime
%   'full': use the original larger grid
%   'minimal': extremely small grid (keep only a few IPT-specific degrees of freedom)
%   'compact': small grid with a few extra degrees of freedom (still lightweight)
%
% Outputs:
%   Investment-potential-tracking/results_fixed_params/ipt_fixed_<objective>_summary.csv
%   Investment-potential-tracking/results_fixed_params/ipt_fixed_<objective>_summary.txt
%   Investment-potential-tracking/results_fixed_params/ipt_fixed_<dataset>.txt

    p = inputParser;
    addParameter(p, 'win_size', 5);
    addParameter(p, 'epsilon', 100);
    addParameter(p, 'tran_cost', 0.001);
    addParameter(p, 'K', 3);
    addParameter(p, 'use_parallel', true);
    addParameter(p, 'num_workers', 4);
    addParameter(p, 'tie_factors', true);
    addParameter(p, 'val_objective', 'log_wealth'); % 'log_wealth' | 'calmar' | 'log_wealth_turnover' | 'log_wealth_q25' | 'log_wealth_min' | 'log_wealth_last' | 'log_wealth_recent' | 'log_wealth_stable' | 'log_wealth_plus_sharpe' | 'rank_wealth_sharpe' | 'log_then_sharpe' | 'log_then_calmar' | 'log_then_turnover' | 'log_then_turnover_constrained' | 'log_then_wins_both' | 'wins_both'
    addParameter(p, 'two_stage_topN', 20); % used when val_objective='log_then_calmar' or 'log_then_turnover'
    addParameter(p, 'two_stage_turnover_quantile', 0.25); % used when val_objective='log_then_turnover_constrained'
    addParameter(p, 'turnover_penalty_lambda', 0.01);
    addParameter(p, 'val_log_wealth_cap', Inf); % cap log(fold_terminal_wealth) during validation scoring (Inf disables)
    addParameter(p, 'val_sharpe_weight', 0); % weight for 'log_wealth_plus_sharpe' (0 reduces to plain log_wealth)
    addParameter(p, 'sharpe_annualization', 252);
    addParameter(p, 'Q_clip_max', Inf); % clip Q_factor to [-Q_clip_max, Q_clip_max]
    addParameter(p, 'Q_clip_max_values', []); % optional list for grid search (Inf allowed)
    addParameter(p, 'run_tag', ''); % optional suffix to avoid overwriting outputs
    addParameter(p, 'xrel_clip_mode', 'none'); % 'none' | 'fixed' | 'percentile'
    addParameter(p, 'xrel_clip_fixed', [0.5, 1.5]); % [lo, hi] when mode='fixed'
    addParameter(p, 'xrel_clip_prc', [0.5, 99.5]); % [p_lo, p_hi] when mode='percentile'
    addParameter(p, 'xrel_extreme_topk', 5); % report top-k min/max per dataset (0 disables)
    % NOTE: active_function.m in this repo currently implements the original hard gating only.
    addParameter(p, 'gating_mode', 'hard'); % kept for backward compatibility; must be 'hard'
    addParameter(p, 'trend_win', 21); % unused
    addParameter(p, 'trend_gamma', 5); % unused
    addParameter(p, 'risk_sigma_factor', 0.15); % unused
    addParameter(p, 'trend_guard_reversal', false); % unused
    addParameter(p, 'Q_clip_highrisk', Inf); % unused
    addParameter(p, 'extreme_confirm_days', 3); % confirm days for extreme high-risk (state5) before using 2*risk_factor
    addParameter(p, 'extreme_confirm_days_values', []); % optional list for grid search
    addParameter(p, 'high_confirm_days', 1); % confirm days for high-risk (state4/5) before using risk_factor
    addParameter(p, 'high_confirm_days_values', []); % optional list for grid search
    addParameter(p, 'near_risk_mode', 'by_weight'); % 'by_weight' (default) | 'by_risk'
    addParameter(p, 'risk_threshold_mode', 'scale'); % 'scale' (default) | 'near_prc_fixed' | 'near_prc_from_q'
    addParameter(p, 'near_risk_prc_high', 80); % used when risk_threshold_mode='near_prc_fixed'
    addParameter(p, 'near_risk_prc_extreme', 95); % used when risk_threshold_mode='near_prc_fixed'
    addParameter(p, 'state_adaptive_trading', false); % make update_mix/max_turnover depend on risk state weights
    addParameter(p, 'risk_high_floor', 0.25); % floor for risk_high when loosening max_turnover (used when state_adaptive_trading=true)
    addParameter(p, 'update_mix', 1); % inertia in (0,1], smaller reduces turnover
    addParameter(p, 'adaptive_inertia_q', false); % IPT-ADC: scale update by 1/(1+|Q_t|)
    addParameter(p, 'update_mix_values', []); % optional list for grid search (e.g. [0.2, 0.5, 1])
    addParameter(p, 'max_turnover', Inf); % per-step turnover cap on weight update (Inf disables)
    addParameter(p, 'max_turnover_values', []); % optional list for grid search
    addParameter(p, 'grid_profile', 'robust'); % 'robust' | 'minimal' | 'compact' | 'full'
    addParameter(p, 'datasets', []); % [] for all, or e.g. {'ndx','tse'} or "ndx"
    addParameter(p, 'train_ratio', 0.6); % unused when split_mode='dev_test'
    addParameter(p, 'val_ratio', 0.2);   % unused when split_mode='dev_test'
    addParameter(p, 'split_mode', 'dev_test'); % only support dev_test now
    addParameter(p, 'dev_ratio', 0.6);
    addParameter(p, 'tune_recent_len', Inf); % use only the most recent part of the scored tuning segment (Inf disables)
    addParameter(p, 'L_smoothing_alpha', 0.2);
    addParameter(p, 'L_percentiles', [92.5, 95, 97.5]);
    addParameter(p, 'weight_inspect_wins_list', [63, 126, 252]);
    addParameter(p, 'risk_inspect_wins_list', [21, 42]);
    % q controls threshold scaling in active_function ((1-q)*L_near, (1-q/2)*L_near).
    % Large q makes high-risk too frequent; keep q in a tail-like regime by default.
    addParameter(p, 'q_values', [0.05, 0.10, 0.15, 0.20]);
    addParameter(p, 'factor_values', [5, 10, 20, 50]);
    parse(p, varargin{:});
    opts = p.Results;

    val_objective = lower(string(opts.val_objective));
    if val_objective ~= "log_wealth" && val_objective ~= "calmar" && val_objective ~= "log_wealth_turnover" && ...
            val_objective ~= "log_wealth_q25" && val_objective ~= "log_wealth_min" && val_objective ~= "log_wealth_last" && val_objective ~= "log_wealth_recent" && ...
            val_objective ~= "log_wealth_stable" && ...
            val_objective ~= "log_wealth_plus_sharpe" && ...
            val_objective ~= "rank_wealth_sharpe" && val_objective ~= "log_then_sharpe" && val_objective ~= "log_then_calmar" && val_objective ~= "log_then_turnover" && val_objective ~= "log_then_turnover_constrained" && ...
            val_objective ~= "log_then_wins_both" && val_objective ~= "wins_both"
        error('Unsupported val_objective: %s (use log_wealth, calmar, log_wealth_turnover, log_wealth_q25, log_wealth_min, log_wealth_last, log_wealth_recent, log_wealth_stable, log_wealth_plus_sharpe, rank_wealth_sharpe, log_then_sharpe, log_then_calmar, log_then_turnover, log_then_turnover_constrained, log_then_wins_both, or wins_both)', val_objective);
    end
    if ~(isnumeric(opts.sharpe_annualization) && isscalar(opts.sharpe_annualization) && opts.sharpe_annualization > 0 && isfinite(opts.sharpe_annualization))
        error('sharpe_annualization must be a positive finite scalar.');
    end
    if ~(isnumeric(opts.two_stage_topN) && isscalar(opts.two_stage_topN) && opts.two_stage_topN >= 1 && mod(opts.two_stage_topN, 1) == 0)
        error('two_stage_topN must be a positive integer.');
    end
    if ~(isnumeric(opts.two_stage_turnover_quantile) && isscalar(opts.two_stage_turnover_quantile) && opts.two_stage_turnover_quantile > 0 && opts.two_stage_turnover_quantile <= 1)
        error('two_stage_turnover_quantile must be in (0,1].');
    end
    if opts.turnover_penalty_lambda < 0
        error('turnover_penalty_lambda must be >= 0');
    end
    if ~(isnumeric(opts.val_log_wealth_cap) && isscalar(opts.val_log_wealth_cap) && opts.val_log_wealth_cap > 0)
        error('val_log_wealth_cap must be a positive scalar (use Inf to disable).');
    end
    if ~(isnumeric(opts.val_sharpe_weight) && isscalar(opts.val_sharpe_weight) && isfinite(opts.val_sharpe_weight) && opts.val_sharpe_weight >= 0)
        error('val_sharpe_weight must be a finite scalar >= 0.');
    end
    if ~(isnumeric(opts.tune_recent_len) && isscalar(opts.tune_recent_len) && opts.tune_recent_len > 0)
        error('tune_recent_len must be a positive scalar (use Inf to disable).');
    end
    if ~(isnumeric(opts.Q_clip_max) && isscalar(opts.Q_clip_max) && opts.Q_clip_max > 0)
        error('Q_clip_max must be a positive scalar (use Inf to disable clipping)');
    end
    if ~(isempty(opts.Q_clip_max_values) || (isnumeric(opts.Q_clip_max_values) && all(~isnan(opts.Q_clip_max_values)) && all(opts.Q_clip_max_values > 0)))
        error('Q_clip_max_values must be empty or a numeric vector of positive values (Inf allowed).');
    end
    if ~(isnumeric(opts.update_mix) && isscalar(opts.update_mix) && opts.update_mix > 0 && opts.update_mix <= 1)
        error('update_mix must be in (0,1].');
    end
    if ~(isempty(opts.update_mix_values) || (isnumeric(opts.update_mix_values) && all(isfinite(opts.update_mix_values)) && all(opts.update_mix_values > 0) && all(opts.update_mix_values <= 1)))
        error('update_mix_values must be empty or a numeric vector within (0,1].');
    end
    if ~(isnumeric(opts.max_turnover) && isscalar(opts.max_turnover) && opts.max_turnover > 0)
        error('max_turnover must be a positive scalar (use Inf to disable).');
    end
    if ~(isempty(opts.max_turnover_values) || (isnumeric(opts.max_turnover_values) && all(~isnan(opts.max_turnover_values)) && all(opts.max_turnover_values > 0)))
        error('max_turnover_values must be empty or a numeric vector of positive values (Inf allowed).');
    end
    xrel_clip_mode = lower(string(opts.xrel_clip_mode));
    if xrel_clip_mode ~= "none" && xrel_clip_mode ~= "fixed" && xrel_clip_mode ~= "percentile"
        error('xrel_clip_mode must be one of: none, fixed, percentile');
    end
    if ~(isnumeric(opts.xrel_clip_fixed) && numel(opts.xrel_clip_fixed) == 2)
        error('xrel_clip_fixed must be a 2-element numeric vector [lo, hi]');
    end
    if ~(isnumeric(opts.xrel_clip_prc) && numel(opts.xrel_clip_prc) == 2 && opts.xrel_clip_prc(1) >= 0 && opts.xrel_clip_prc(2) <= 100 && opts.xrel_clip_prc(1) < opts.xrel_clip_prc(2))
        error('xrel_clip_prc must be a 2-element percentile vector [p_lo, p_hi] within [0,100]');
    end
    if ~(isnumeric(opts.xrel_extreme_topk) && isscalar(opts.xrel_extreme_topk) && opts.xrel_extreme_topk >= 0)
        error('xrel_extreme_topk must be >= 0');
    end
    gating_mode = lower(string(opts.gating_mode));
    if gating_mode ~= "hard"
        error('Only gating_mode=hard is supported in this codebase (active_function.m).');
    end
    if ~(isnumeric(opts.trend_win) && isscalar(opts.trend_win) && opts.trend_win >= 1)
        error('trend_win must be a positive scalar.');
    end
    if ~(isnumeric(opts.trend_gamma) && isscalar(opts.trend_gamma) && isfinite(opts.trend_gamma))
        error('trend_gamma must be a finite scalar.');
    end
    if ~(isnumeric(opts.risk_sigma_factor) && isscalar(opts.risk_sigma_factor) && opts.risk_sigma_factor > 0)
        error('risk_sigma_factor must be a positive scalar.');
    end
    if ~(isnumeric(opts.extreme_confirm_days) && isscalar(opts.extreme_confirm_days) && isfinite(opts.extreme_confirm_days) && opts.extreme_confirm_days >= 1 && mod(opts.extreme_confirm_days, 1) == 0)
        error('extreme_confirm_days must be a positive integer.');
    end
    if ~(isempty(opts.extreme_confirm_days_values) || (isnumeric(opts.extreme_confirm_days_values) && all(isfinite(opts.extreme_confirm_days_values)) && all(opts.extreme_confirm_days_values >= 1) && all(mod(opts.extreme_confirm_days_values, 1) == 0)))
        error('extreme_confirm_days_values must be empty or a vector of positive integers.');
    end
    if ~(isnumeric(opts.high_confirm_days) && isscalar(opts.high_confirm_days) && isfinite(opts.high_confirm_days) && opts.high_confirm_days >= 1 && mod(opts.high_confirm_days, 1) == 0)
        error('high_confirm_days must be a positive integer.');
    end
    if ~(isempty(opts.high_confirm_days_values) || (isnumeric(opts.high_confirm_days_values) && all(isfinite(opts.high_confirm_days_values)) && all(opts.high_confirm_days_values >= 1) && all(mod(opts.high_confirm_days_values, 1) == 0)))
        error('high_confirm_days_values must be empty or a vector of positive integers.');
    end
    near_risk_mode = lower(string(opts.near_risk_mode));
    if near_risk_mode ~= "by_weight" && near_risk_mode ~= "by_risk"
        error('near_risk_mode must be by_weight or by_risk.');
    end
    risk_threshold_mode = lower(string(opts.risk_threshold_mode));
    if risk_threshold_mode ~= "scale" && risk_threshold_mode ~= "near_prc_fixed" && risk_threshold_mode ~= "near_prc_from_q"
        error('risk_threshold_mode must be scale, near_prc_fixed, or near_prc_from_q.');
    end
    if (risk_threshold_mode == "near_prc_fixed" || risk_threshold_mode == "near_prc_from_q") && near_risk_mode ~= "by_risk"
        error('For risk_threshold_mode=%s, please set near_risk_mode=by_risk (to ensure near-risk thresholds are well-defined and cacheable).', risk_threshold_mode);
    end
    if risk_threshold_mode == "near_prc_fixed"
        if ~(isnumeric(opts.near_risk_prc_high) && isscalar(opts.near_risk_prc_high) && opts.near_risk_prc_high >= 0 && opts.near_risk_prc_high <= 100)
            error('near_risk_prc_high must be in [0,100].');
        end
        if ~(isnumeric(opts.near_risk_prc_extreme) && isscalar(opts.near_risk_prc_extreme) && opts.near_risk_prc_extreme >= 0 && opts.near_risk_prc_extreme <= 100)
            error('near_risk_prc_extreme must be in [0,100].');
        end
        if ~(opts.near_risk_prc_extreme >= opts.near_risk_prc_high)
            error('near_risk_prc_extreme must be >= near_risk_prc_high.');
        end
    end
    if ~(islogical(opts.state_adaptive_trading) && isscalar(opts.state_adaptive_trading))
        error('state_adaptive_trading must be a logical scalar.');
    end
    if opts.state_adaptive_trading
        error('state_adaptive_trading=true is not supported with the current active_function.m (no state_meta output).');
    end
    if ~(isnumeric(opts.Q_clip_highrisk) && isscalar(opts.Q_clip_highrisk))
        error('Q_clip_highrisk must be a scalar.');
    end
    if ~(isnumeric(opts.risk_high_floor) && isscalar(opts.risk_high_floor) && opts.risk_high_floor > 0 && opts.risk_high_floor <= 1)
        error('risk_high_floor must be in (0,1].');
    end

    grid_profile = lower(string(opts.grid_profile));
    if grid_profile ~= "robust" && grid_profile ~= "full" && grid_profile ~= "minimal" && grid_profile ~= "compact"
        error('Unsupported grid_profile: %s (use robust, minimal, compact, or full)', grid_profile);
    end

    % Apply a smaller, more robust default grid unless the user explicitly overrides fields.
    if grid_profile == "robust"
        using_defaults = string(p.UsingDefaults);
        if any(using_defaults == "L_percentiles")
            opts.L_percentiles = 92.5; % fixed
        end
        if any(using_defaults == "weight_inspect_wins_list")
            opts.weight_inspect_wins_list = [63, 126, 252];
        end
        if any(using_defaults == "risk_inspect_wins_list")
            opts.risk_inspect_wins_list = 21; % fixed
        end
        if any(using_defaults == "q_values")
            opts.q_values = [0.05, 0.10, 0.15, 0.20];
        end
        if any(using_defaults == "factor_values")
            % Keep reverse/risk factors open (tie_factors controls whether they are tied)
            opts.factor_values = [5, 10, 20, 50];
        end
        if any(using_defaults == "Q_clip_max")
            opts.Q_clip_max = 10;
        end
        if any(using_defaults == "extreme_confirm_days_values")
            opts.extreme_confirm_days_values = [1, 3, 5];
        end
        if any(using_defaults == "high_confirm_days_values")
            opts.high_confirm_days_values = [1, 2, 3];
        end
    elseif grid_profile == "minimal"
        % Extremely small grid: only tune (max_turnover, reverse_factor) with tied factors.
        using_defaults = string(p.UsingDefaults);
        if any(using_defaults == "tie_factors")
            opts.tie_factors = true;
        end
        if any(using_defaults == "L_percentiles")
            opts.L_percentiles = 92.5;
        end
        if any(using_defaults == "weight_inspect_wins_list")
            opts.weight_inspect_wins_list = 252;
        end
        if any(using_defaults == "risk_inspect_wins_list")
            opts.risk_inspect_wins_list = 21;
        end
        if any(using_defaults == "q_values")
            opts.q_values = 0.4;
        end
        if any(using_defaults == "factor_values")
            opts.factor_values = [5, 10, 20, 50];
        end
        if any(using_defaults == "Q_clip_max")
            opts.Q_clip_max = 10;
        end
        if any(using_defaults == "Q_clip_max_values")
            opts.Q_clip_max_values = [];
        end
        if any(using_defaults == "max_turnover_values")
            opts.max_turnover_values = [0.1, 0.5, 2, Inf];
        end
        if any(using_defaults == "max_turnover")
            opts.max_turnover = Inf;
        end
    elseif grid_profile == "compact"
        % Small but less restrictive grid: tune (max_turnover, reverse_factor, q_value, Q_clip_max, weight window).
        % Keep L_percentile and risk window fixed to reduce degrees of freedom.
        using_defaults = string(p.UsingDefaults);
        if any(using_defaults == "tie_factors")
            opts.tie_factors = true;
        end
        if any(using_defaults == "L_percentiles")
            opts.L_percentiles = 92.5;
        end
        if any(using_defaults == "weight_inspect_wins_list")
            opts.weight_inspect_wins_list = [126, 252];
        end
        if any(using_defaults == "risk_inspect_wins_list")
            opts.risk_inspect_wins_list = 21;
        end
        if any(using_defaults == "q_values")
            opts.q_values = [0.3, 0.4, 0.5];
        end
        if any(using_defaults == "factor_values")
            opts.factor_values = [5, 10];
        end
        if any(using_defaults == "Q_clip_max_values")
            opts.Q_clip_max_values = [1, 10, Inf];
        end
        if any(using_defaults == "Q_clip_max")
            opts.Q_clip_max = 10;
        end
        if any(using_defaults == "max_turnover_values")
            opts.max_turnover_values = [0.1, 0.5, 2, Inf];
        end
        if any(using_defaults == "max_turnover")
            opts.max_turnover = Inf;
        end
    end

    script_dir = fileparts(mfilename('fullpath'));
    data_dir = fullfile(script_dir, 'Data Set');
    out_dir = fullfile(script_dir, 'results_fixed_params');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    files = dir(fullfile(data_dir, '*.mat'));
    if isempty(files)
        error('No datasets found in %s', data_dir);
    end
    [~, order] = sort({files.name});
    files = files(order);

    if ~isempty(opts.datasets)
        if ischar(opts.datasets) || isstring(opts.datasets)
            wanted = string(opts.datasets);
        else
            wanted = string(opts.datasets(:));
        end
        wanted = lower(erase(wanted, ".mat"));
        keep = false(numel(files), 1);
        for ii = 1:numel(files)
            keep(ii) = any(lower(erase(string(files(ii).name), ".mat")) == wanted);
        end
        files = files(keep);
        if isempty(files)
            error('No datasets matched opts.datasets.');
        end
    end

    summary = struct('dataset', {}, 'T', {}, 'N', {}, ...
        'train_end', {}, 'val_start', {}, 'val_end', {}, 'test_start', {}, 'test_end', {}, ...
        'train_ratio', {}, 'val_ratio', {}, 'test_ratio', {}, ...
        'split_mode', {}, 'dev_ratio', {}, 'dev_end', {}, 'warmup_end', {}, 'tune_start', {}, 'tune_end', {}, ...
        'K', {}, 'tran_cost', {}, 'win_size', {}, 'epsilon', {}, 'sharpe_annualization', {}, 'update_mix', {}, 'max_turnover', {}, 'Q_clip_max', {}, ...
        'state_adaptive_trading', {}, 'risk_high_floor', {}, 'gating_mode', {}, 'trend_win', {}, 'trend_gamma', {}, 'risk_sigma_factor', {}, ...
        'val_objective', {}, 'val_score', {}, 'val_geom', {}, 'test_wealth', {}, ...
        'primary_log_wealth', {}, 'val_calmar', {}, 'val_turnover_mean', {}, 'val_turnover_threshold', {}, 'val_sharpe', {}, ...
        'weight_inspect_wins', {}, 'risk_inspect_wins', {}, ...
        'L_percentile', {}, 'q_value', {}, 'reverse_factor', {}, 'risk_factor', {}, 'extreme_confirm_days', {}, 'high_confirm_days', {});

    for i = 1:numel(files)
        dataset = erase(files(i).name, '.mat');
        data_path = fullfile(data_dir, files(i).name);
        S = load(data_path, 'data');
        data = S.data;

        % Diagnose/clip extreme daily relatives if requested (helps with unadjusted corporate actions).
        if opts.xrel_extreme_topk > 0
            report_xrel_extremes(out_dir, dataset, data, opts.xrel_extreme_topk, string(opts.run_tag));
        end
        if xrel_clip_mode ~= "none"
            [data, clip_info] = clip_xrel(data, xrel_clip_mode, opts.xrel_clip_fixed, opts.xrel_clip_prc);
            fprintf('NOTE: xrel_clip_mode=%s for %s (lo=%.6g, hi=%.6g, clipped=%d)\n', ...
                xrel_clip_mode, files(i).name, clip_info.lo, clip_info.hi, clip_info.num_clipped);
        end
        [T, N] = size(data);

        split_mode = lower(string(opts.split_mode));
        if split_mode ~= "dev_test"
            error('Unsupported split_mode: %s (now only dev_test is allowed)', split_mode);
        end

        dev = ipt_dev_test_split(T, 'dev_ratio', opts.dev_ratio);
        warmup_len = max([max(opts.weight_inspect_wins_list), max(opts.risk_inspect_wins_list), opts.win_size]);
        score_start = warmup_len + 1;
        score_end = dev.dev_end;
        if score_start > score_end
            error('Dev segment too short after warm-up (T=%d, dev_end=%d, warmup=%d).', T, dev.dev_end, warmup_len);
        end
        split = struct();
        split.train_end = warmup_len; % warm-up only
        split.val_start = score_start; % scored dev remainder
        split.val_end = score_end;
        split.test_start = dev.test_start;
        split.test_end = dev.test_end;
        split.train_ratio = warmup_len / T;
        split.val_ratio = (score_end - score_start + 1) / T;
        split.test_ratio = (dev.test_end - dev.test_start + 1) / T;

        dev_end = dev.dev_end;
        warmup_end = warmup_len;
        tune_start = score_start;
        tune_end = score_end;

        if isfinite(opts.tune_recent_len)
            recent_start = score_end - floor(double(opts.tune_recent_len)) + 1;
            score_start = max(score_start, recent_start);
        end
        split.val_start = score_start;
        split.val_end = score_end;
        if split_mode == "dev_test"
            tune_start = score_start;
            tune_end = score_end;
        end

        K = opts.K;
        val_len_total = score_end - score_start + 1;
        fold_len = floor(val_len_total / K);
        if fold_len < 1
            error('Validation segment too short for K=%d (val_len=%d).', K, val_len_total);
        end
        fold_ranges = zeros(K, 2);
        for k = 1:K
            f_start = score_start + (k - 1) * fold_len;
            if k == K
                f_end = score_end;
            else
                f_end = f_start + fold_len - 1;
            end
            fold_ranges(k, :) = [f_start, f_end];
        end

        p_close = ones(T, N);
        for t = 2:T
            p_close(t, :) = p_close(t - 1, :) .* data(t, :);
        end

        ratio = ubah_price_ratio(data);

        baseline_val_log = [];
        baseline_val_sharpe = [];
        baseline_names = [];
        needs_baseline_wins = (val_objective == "log_then_wins_both" || val_objective == "wins_both");
        if needs_baseline_wins
            fprintf('Precomputing baseline validation metrics (for wins_both)...\n');
            [baseline_val_log, baseline_val_sharpe, baseline_names] = compute_baseline_val_scores( ...
                data, fold_ranges, opts.tran_cost, opts.win_size, opts.epsilon, opts.val_log_wealth_cap, opts.sharpe_annualization);
        end

        weight_list = opts.weight_inspect_wins_list(:)';
        risk_list = opts.risk_inspect_wins_list(:)';
        L_percentiles = opts.L_percentiles(:)';
        q_values = opts.q_values(:)';
        factor_values = opts.factor_values(:)';
        tie_factors = logical(opts.tie_factors);
        if isempty(opts.update_mix_values)
            update_mix_list = opts.update_mix;
        else
            update_mix_list = opts.update_mix_values(:)';
        end
        if isempty(opts.max_turnover_values)
            max_turnover_list = opts.max_turnover;
        else
            max_turnover_list = opts.max_turnover_values(:)';
        end
        if isempty(opts.Q_clip_max_values)
            q_clip_list = opts.Q_clip_max;
        else
            q_clip_list = opts.Q_clip_max_values(:)';
        end
        if isempty(opts.extreme_confirm_days_values)
            extreme_confirm_days_list = double(opts.extreme_confirm_days);
        else
            extreme_confirm_days_list = double(opts.extreme_confirm_days_values(:)');
        end
        if isempty(opts.high_confirm_days_values)
            high_confirm_days_list = double(opts.high_confirm_days);
        else
            high_confirm_days_list = double(opts.high_confirm_days_values(:)');
        end

        num_weights = numel(weight_list);
        num_risks = numel(risk_list);
        num_update_mix = numel(update_mix_list);
        yar_weights_long_cache = cell(num_weights, 1);
        yar_weights_near_cache = cell(num_weights, 1);
        half_weight_cache = zeros(num_weights, 1);

        for wi = 1:num_weights
            w = weight_list(wi);
            half_weight = floor(w / 2);
            yar_weights_long_cache{wi} = yar_weights(data, w);
            yar_weights_near_cache{wi} = yar_weights(data, half_weight);
            half_weight_cache(wi) = half_weight;
        end

        yar_ubah_long_cache = cell(num_weights, num_risks);
        yar_ubah_near_cache = cell(num_weights, num_risks);
        valid_pair = false(num_weights, num_risks);
        for wi = 1:num_weights
            w = weight_list(wi);
            half_weight = half_weight_cache(wi);
            for ri = 1:num_risks
                r = risk_list(ri);
                if r > w
                    continue;
                end
                r3 = max(2, floor(r / 3));
                half_risk = floor(r / 2);
                half_r3 = max(2, floor(half_risk / 3));
                start_long = w - r3 + 1;
                start_near = half_weight - half_r3 + 1;
                if start_long < 1 || start_near < 1
                    continue;
                end
                yar_ubah_long_cache{wi, ri} = yar_ubah(ratio(start_long:T, :), r3);
                if near_risk_mode == "by_weight"
                    yar_ubah_near_cache{wi, ri} = yar_ubah(ratio(start_near:T, :), half_r3);
                else
                    yar_ubah_near_cache{wi, ri} = yar_ubah(ratio, half_r3);
                end
                valid_pair(wi, ri) = true;
            end
        end

        near_L_high_cache = cell(num_risks, numel(q_values));
        near_L_ext_cache = cell(num_risks, numel(q_values));
        if risk_threshold_mode == "near_prc_fixed" || risk_threshold_mode == "near_prc_from_q"
            for ri = 1:num_risks
                if ~any(valid_pair(:, ri))
                    continue;
                end
                % In by_risk mode, yar_ubah_near_cache is identical across wi; use wi=1.
                yar_near_all = yar_ubah_near_cache{1, ri};
                if isempty(yar_near_all)
                    continue;
                end
                yar_near_all = yar_near_all(:, 1);
                for qi = 1:numel(q_values)
                    if risk_threshold_mode == "near_prc_fixed"
                        prc_hi = double(opts.near_risk_prc_high);
                        prc_ext = double(opts.near_risk_prc_extreme);
                    else
                        qv = double(q_values(qi));
                        prc_hi = 100 * (1 - qv);
                        prc_ext = 100 * (1 - qv / 2);
                        prc_hi = max(0, min(100, prc_hi));
                        prc_ext = max(0, min(100, prc_ext));
                    end
                    Lh = compute_yar_percentile(yar_near_all, prc_hi);
                    Le = compute_yar_percentile(yar_near_all, prc_ext);
                    near_L_high_cache{ri, qi} = ipt_smooth_series(Lh, opts.L_smoothing_alpha);
                    near_L_ext_cache{ri, qi} = ipt_smooth_series(Le, opts.L_smoothing_alpha);
                end
            end
        end

        if tie_factors
            [WI, RI, LI, QI, FI, UMI, TI, CI, EDI, HDI] = ndgrid( ...
                1:num_weights, 1:num_risks, 1:numel(L_percentiles), 1:numel(q_values), 1:numel(factor_values), 1:num_update_mix, 1:numel(max_turnover_list), 1:numel(q_clip_list), 1:numel(extreme_confirm_days_list), 1:numel(high_confirm_days_list));
            combos = [WI(:), RI(:), LI(:), QI(:), FI(:), UMI(:), TI(:), CI(:), EDI(:), HDI(:)];
        else
            [WI, RI, LI, QI, REVI, RFI, UMI, TI, CI, EDI, HDI] = ndgrid( ...
                1:num_weights, 1:num_risks, 1:numel(L_percentiles), 1:numel(q_values), 1:numel(factor_values), 1:numel(factor_values), 1:num_update_mix, 1:numel(max_turnover_list), 1:numel(q_clip_list), 1:numel(extreme_confirm_days_list), 1:numel(high_confirm_days_list));
            combos = [WI(:), RI(:), LI(:), QI(:), REVI(:), RFI(:), UMI(:), TI(:), CI(:), EDI(:), HDI(:)];
        end
        pair_ok = valid_pair(sub2ind(size(valid_pair), combos(:, 1), combos(:, 2)));
        combos = combos(pair_ok, :);
        if isempty(combos)
            error('No valid parameter combinations after window constraints.');
        end

        num_combos = size(combos, 1);
        needs_multi_scores = (val_objective == "log_then_calmar" || val_objective == "log_then_turnover" || val_objective == "log_then_turnover_constrained" || val_objective == "rank_wealth_sharpe" || val_objective == "log_then_sharpe" || val_objective == "log_then_wins_both" || val_objective == "wins_both");
        if needs_multi_scores
            scores_log = -inf(num_combos, 1);
            scores_calmar = -inf(num_combos, 1);
            turnover_means = inf(num_combos, 1);
            scores_sharpe = -inf(num_combos, 1);
            wins_both_counts = -inf(num_combos, 1);
        else
            scores = -inf(num_combos, 1);
        end

        fprintf('\n=== IPT fixed (win_size=%d, epsilon=%.1f) : %s ===\n', opts.win_size, opts.epsilon, files(i).name);
        fprintf('split_mode=%s, Objective=%s, grid_profile=%s, K=%d, grid=%d\n', split_mode, val_objective, grid_profile, K, num_combos);

        use_parallel = opts.use_parallel && ~isempty(ver('parallel'));
        if use_parallel
            try
                pool = gcp('nocreate');
                if isempty(pool)
                    parpool(opts.num_workers);
                end
            catch
                use_parallel = false;
            end
        end

        if use_parallel
            if needs_multi_scores
                parfor idx = 1:num_combos
                    [s_log, s_cal, tmean, s_sh] = eval_combo(idx, combos, tie_factors, weight_list, risk_list, L_percentiles, q_values, factor_values, update_mix_list, max_turnover_list, q_clip_list, extreme_confirm_days_list, high_confirm_days_list, near_risk_mode, risk_threshold_mode, near_L_high_cache, near_L_ext_cache, ...
                        yar_weights_long_cache, yar_weights_near_cache, yar_ubah_long_cache, yar_ubah_near_cache, ...
                        data, p_close, fold_ranges, opts.tran_cost, opts.win_size, opts.epsilon, opts.L_smoothing_alpha, "log_wealth", opts.turnover_penalty_lambda, opts.val_log_wealth_cap, opts.val_sharpe_weight, opts.update_mix, ...
                        gating_mode, opts.trend_win, opts.trend_gamma, opts.risk_sigma_factor, opts.trend_guard_reversal, opts.Q_clip_highrisk, opts.state_adaptive_trading, opts.risk_high_floor, opts.adaptive_inertia_q, opts.sharpe_annualization);
                    scores_log(idx) = s_log;
                    scores_calmar(idx) = s_cal;
                    turnover_means(idx) = tmean;
                    scores_sharpe(idx) = s_sh;
                end
            else
                parfor idx = 1:num_combos
                    score = eval_combo(idx, combos, tie_factors, weight_list, risk_list, L_percentiles, q_values, factor_values, update_mix_list, max_turnover_list, q_clip_list, extreme_confirm_days_list, high_confirm_days_list, near_risk_mode, risk_threshold_mode, near_L_high_cache, near_L_ext_cache, ...
                        yar_weights_long_cache, yar_weights_near_cache, yar_ubah_long_cache, yar_ubah_near_cache, ...
                        data, p_close, fold_ranges, opts.tran_cost, opts.win_size, opts.epsilon, opts.L_smoothing_alpha, val_objective, opts.turnover_penalty_lambda, opts.val_log_wealth_cap, opts.val_sharpe_weight, opts.update_mix, ...
                        gating_mode, opts.trend_win, opts.trend_gamma, opts.risk_sigma_factor, opts.trend_guard_reversal, opts.Q_clip_highrisk, opts.state_adaptive_trading, opts.risk_high_floor, opts.adaptive_inertia_q, opts.sharpe_annualization);
                    scores(idx) = score;
                end
            end
        else
            if needs_multi_scores
                for idx = 1:num_combos
                    [scores_log(idx), scores_calmar(idx), turnover_means(idx), scores_sharpe(idx)] = eval_combo(idx, combos, tie_factors, weight_list, risk_list, L_percentiles, q_values, factor_values, update_mix_list, max_turnover_list, q_clip_list, extreme_confirm_days_list, high_confirm_days_list, near_risk_mode, risk_threshold_mode, near_L_high_cache, near_L_ext_cache, ...
                        yar_weights_long_cache, yar_weights_near_cache, yar_ubah_long_cache, yar_ubah_near_cache, ...
                        data, p_close, fold_ranges, opts.tran_cost, opts.win_size, opts.epsilon, opts.L_smoothing_alpha, "log_wealth", opts.turnover_penalty_lambda, opts.val_log_wealth_cap, opts.val_sharpe_weight, opts.update_mix, ...
                        gating_mode, opts.trend_win, opts.trend_gamma, opts.risk_sigma_factor, opts.state_adaptive_trading, opts.risk_high_floor, opts.sharpe_annualization);
                end
            else
                for idx = 1:num_combos
                    scores(idx) = eval_combo(idx, combos, tie_factors, weight_list, risk_list, L_percentiles, q_values, factor_values, update_mix_list, max_turnover_list, q_clip_list, extreme_confirm_days_list, high_confirm_days_list, near_risk_mode, risk_threshold_mode, near_L_high_cache, near_L_ext_cache, ...
                        yar_weights_long_cache, yar_weights_near_cache, yar_ubah_long_cache, yar_ubah_near_cache, ...
                        data, p_close, fold_ranges, opts.tran_cost, opts.win_size, opts.epsilon, opts.L_smoothing_alpha, val_objective, opts.turnover_penalty_lambda, opts.val_log_wealth_cap, opts.val_sharpe_weight, opts.update_mix, ...
                        gating_mode, opts.trend_win, opts.trend_gamma, opts.risk_sigma_factor, opts.trend_guard_reversal, opts.Q_clip_highrisk, opts.state_adaptive_trading, opts.risk_high_floor, opts.adaptive_inertia_q, opts.sharpe_annualization);
                end
            end
        end

        if needs_baseline_wins
            wins_both_counts = compute_wins_both_counts(scores_log, scores_sharpe, baseline_val_log, baseline_val_sharpe);
            if all(~isfinite(wins_both_counts))
                warning('wins_both_counts all non-finite; check baseline precompute & sharpe computation.');
            end
        end

        if val_objective == "log_then_calmar"
            topN = min(double(opts.two_stage_topN), num_combos);
            [~, ord] = sort(scores_log, 'descend');
            cand = ord(1:topN);
            [best_score, ii] = max(scores_calmar(cand));
            best_idx = cand(ii);
            fprintf('Two-stage selection: topN=%d by log_wealth, then best by calmar.\n', topN);
        elseif val_objective == "log_then_turnover"
            topN = min(double(opts.two_stage_topN), num_combos);
            [~, ord] = sort(scores_log, 'descend');
            cand = ord(1:topN);
            keys = [turnover_means(cand), -scores_log(cand)];
            [~, ii] = sortrows(keys, [1, 2]);
            best_idx = cand(ii(1));
            best_score = -turnover_means(best_idx);
            fprintf('Two-stage selection: topN=%d by log_wealth, then min turnover.\n', topN);
        elseif val_objective == "log_then_turnover_constrained"
            topN = min(double(opts.two_stage_topN), num_combos);
            [~, ord] = sort(scores_log, 'descend');
            cand = ord(1:topN);
            tvals = turnover_means(cand);
            q = double(opts.two_stage_turnover_quantile);
            thr = quantile_base(tvals, q);
            finite_t = tvals(isfinite(tvals));
            if isempty(finite_t)
                tol = 0;
            else
                tol = 1e-12 * max(1, max(abs(finite_t)));
            end
            keep = cand(isfinite(tvals) & tvals <= thr + tol);
            if isempty(keep)
                keep = cand(isfinite(tvals));
            end
            if isempty(keep)
                best_idx = cand(1);
            else
                keys = [-scores_log(keep), turnover_means(keep)];
                [~, ii] = sortrows(keys, [1, 2]);
                best_idx = keep(ii(1));
            end
            best_score = scores_log(best_idx);
            fprintf('Two-stage selection: topN=%d by log_wealth, then max log_wealth with turnover <= q%.0f%% threshold.\n', topN, 100*q);
        elseif val_objective == "rank_wealth_sharpe"
            ranks_log = rank_desc(scores_log);
            ranks_sh = rank_desc(scores_sharpe);
            composite = -(ranks_log + ranks_sh);
            [best_score, best_idx] = max(composite);
            fprintf('Dual-objective selection: max -(rank(log_wealth)+rank(sharpe)).\n');
        elseif val_objective == "log_then_sharpe"
            topN = min(double(opts.two_stage_topN), num_combos);
            [~, ord] = sort(scores_log, 'descend');
            cand = ord(1:topN);
            keys = [-scores_sharpe(cand), -scores_log(cand)];
            [~, ii] = sortrows(keys, [1, 2]);
            best_idx = cand(ii(1));
            best_score = scores_sharpe(best_idx);
            fprintf('Two-stage selection: topN=%d by log_wealth, then best by sharpe (tie-break by log_wealth).\n', topN);
        elseif val_objective == "log_then_wins_both"
            topN = min(double(opts.two_stage_topN), num_combos);
            [~, ord] = sort(scores_log, 'descend');
            cand = ord(1:topN);
            keys = [-wins_both_counts(cand), -scores_log(cand), -scores_sharpe(cand)];
            [~, ii] = sortrows(keys, [1, 2, 3]);
            best_idx = cand(ii(1));
            best_score = wins_both_counts(best_idx);
            fprintf('Two-stage selection: topN=%d by log_wealth, then best by wins_both (tie-break by log_wealth, then sharpe).\n', topN);
        elseif val_objective == "wins_both"
            keys = [-wins_both_counts, -scores_log, -scores_sharpe];
            [~, ii] = sortrows(keys, [1, 2, 3]);
            best_idx = ii(1);
            best_score = wins_both_counts(best_idx);
            fprintf('Dual-objective selection: best by wins_both (tie-break by log_wealth, then sharpe).\n');
        else
            [best_score, best_idx] = max(scores);
        end
        best = struct();
        best.val_objective = char(val_objective);
        best.val_score = best_score;
        if exist('scores_sharpe', 'var') && ~isempty(scores_sharpe)
            best.val_sharpe = scores_sharpe(best_idx);
        end
        if val_objective == "log_then_calmar"
            best.primary_log_wealth = scores_log(best_idx);
            best.val_calmar = scores_calmar(best_idx);
        elseif val_objective == "log_then_turnover"
            best.primary_log_wealth = scores_log(best_idx);
            best.val_turnover_mean = turnover_means(best_idx);
            best.val_calmar = scores_calmar(best_idx);
        elseif val_objective == "log_then_turnover_constrained"
            cand = ord(1:min(double(opts.two_stage_topN), num_combos));
            best.primary_log_wealth = scores_log(best_idx);
            best.val_turnover_mean = turnover_means(best_idx);
            best.val_calmar = scores_calmar(best_idx);
            best.val_turnover_threshold = quantile_base(turnover_means(cand), double(opts.two_stage_turnover_quantile));
        elseif val_objective == "rank_wealth_sharpe"
            best.primary_log_wealth = scores_log(best_idx);
            best.val_calmar = scores_calmar(best_idx);
        elseif val_objective == "log_then_sharpe"
            best.primary_log_wealth = scores_log(best_idx);
            best.val_calmar = scores_calmar(best_idx);
        elseif val_objective == "log_then_wins_both" || val_objective == "wins_both"
            best.primary_log_wealth = scores_log(best_idx);
            best.val_calmar = scores_calmar(best_idx);
            best.val_wins_both = wins_both_counts(best_idx);
            if ~isempty(baseline_names)
                best.val_wins_both_outof = numel(baseline_names);
            end
        end

        wi = combos(best_idx, 1);
        ri = combos(best_idx, 2);
        li = combos(best_idx, 3);
        qi = combos(best_idx, 4);

        best.weight_inspect_wins = weight_list(wi);
        best.risk_inspect_wins = risk_list(ri);
        best.L_percentile = L_percentiles(li);
        best.q_value = q_values(qi);

        if tie_factors
            umi = combos(best_idx, 6);
            ti = combos(best_idx, 7);
            ci = combos(best_idx, 8);
            edi = combos(best_idx, 9);
            hdi = combos(best_idx, 10);
        else
            umi = combos(best_idx, 7);
            ti = combos(best_idx, 8);
            ci = combos(best_idx, 9);
            edi = combos(best_idx, 10);
            hdi = combos(best_idx, 11);
        end
        best.update_mix = update_mix_list(umi);
        best.max_turnover = max_turnover_list(ti);
        best.Q_clip_max = q_clip_list(ci);
        best.extreme_confirm_days = extreme_confirm_days_list(edi);
        best.high_confirm_days = high_confirm_days_list(hdi);

        if tie_factors
            fi = combos(best_idx, 5);
            best.reverse_factor = factor_values(fi);
            best.risk_factor = best.reverse_factor;
        else
            revi = combos(best_idx, 5);
            rfi = combos(best_idx, 6);
            best.reverse_factor = factor_values(revi);
            best.risk_factor = factor_values(rfi);
        end

        yar_weights_long = yar_weights_long_cache{wi};
        yar_weights_near = yar_weights_near_cache{wi};
        yar_ubah_long = yar_ubah_long_cache{wi, ri};
        yar_ubah_near = yar_ubah_near_cache{wi, ri};

        L_long_raw = compute_yar_percentile(yar_ubah_long(:, 1), best.L_percentile);
        L_long_history = ipt_smooth_series(L_long_raw, opts.L_smoothing_alpha);
        L_near_raw = compute_yar_percentile(yar_ubah_near(:, 1), best.L_percentile);
        L_near_history = ipt_smooth_series(L_near_raw, opts.L_smoothing_alpha);

        near_L_high = [];
        near_L_ext = [];
        risk_threshold_mode_here = lower(string(opts.risk_threshold_mode));
        if risk_threshold_mode_here ~= "scale"
            near_L_high = near_L_high_cache{ri, qi};
            near_L_ext = near_L_ext_cache{ri, qi};
        end

        [w_YAR, Q_factor] = active_function( ...
            yar_weights_long, yar_weights_near, ...
            yar_ubah_long, yar_ubah_near, ...
            data, best.weight_inspect_wins, ...
            best.reverse_factor, best.risk_factor, best.q_value, L_long_history, L_near_history);
        state_meta = [];
        Q_factor = clip_q(Q_factor, best.Q_clip_max);

        update_mix_used = best.update_mix;
        max_turnover_used = best.max_turnover;
        if opts.state_adaptive_trading && ~isempty(state_meta) && isfield(state_meta, 'g_weights')
            gw = state_meta.g_weights; % (T x 5)
            risk_high = sum(gw(:, 4:5), 2);
            risk_high(~isfinite(risk_high)) = 0;

            % More aggressive when not in high-risk: alpha_t in [update_mix, 1].
            update_mix_used = 1 - (1 - best.update_mix) .* risk_high;
            update_mix_used = max(1e-6, min(1, update_mix_used));

            if isinf(best.max_turnover)
                max_turnover_used = Inf;
            else
                denom = max(double(opts.risk_high_floor), risk_high);
                max_turnover_used = best.max_turnover ./ denom; % loosen in non-high-risk, keep base in high-risk
                max_turnover_used(~isfinite(max_turnover_used)) = best.max_turnover;
            end
        end

        fold_wealths = zeros(K, 1);
        fold_mdds = zeros(K, 1);
        fold_turnovers = zeros(K, 1);
        for k = 1:K
            [fold_wealths(k), fold_mdds(k), fold_turnovers(k)] = eval_ipt_segment(data, p_close, w_YAR, Q_factor, ...
                opts.win_size, opts.tran_cost, opts.epsilon, ...
                fold_ranges(k, 1), fold_ranges(k, 2), update_mix_used, max_turnover_used, opts.sharpe_annualization, opts.adaptive_inertia_q);
        end
        best.val_wealth_folds = fold_wealths;
        best.val_geom = exp(mean(log(max(fold_wealths, realmin))));

        test_wealth = eval_ipt_segment(data, p_close, w_YAR, Q_factor, ...
            opts.win_size, opts.tran_cost, opts.epsilon, split.test_start, split.test_end, update_mix_used, max_turnover_used, opts.sharpe_annualization, opts.adaptive_inertia_q);

        if lower(string(opts.split_mode)) == "dev_test"
            split_tag = sprintf('dev%.0f_test%.0f', 100 * opts.dev_ratio, 100 * (1 - opts.dev_ratio));
        else
            split_tag = sprintf('%.0f_%.0f_%.0f', 100 * opts.train_ratio, 100 * opts.val_ratio, 100 * (1 - opts.train_ratio - opts.val_ratio));
        end
        clip_tag = '';
        if ~isinf(opts.Q_clip_max)
            clip_tag = sprintf('_Qclip%.4g', opts.Q_clip_max);
        end
        if val_objective == "log_wealth_turnover"
            obj_tag = sprintf('%s_lambda%.4g%s', char(val_objective), opts.turnover_penalty_lambda, clip_tag);
        else
            obj_tag = sprintf('%s%s', char(val_objective), clip_tag);
        end
        run_tag = string(opts.run_tag);
        if strlength(run_tag) > 0
            run_tag = "_" + run_tag;
        end
        txt_path = fullfile(out_dir, sprintf('ipt_fixed_%s_%s_%s_%s%s.txt', dataset, obj_tag, split_tag, grid_profile, run_tag));
        fid = fopen(txt_path, 'w');
        if fid ~= -1
            fprintf(fid, 'dataset=%s\n', dataset);
            fprintf(fid, 'T=%d, N=%d\n', T, N);
            fprintf(fid, 'split_mode=%s\n', split_mode);
            if split_mode == "dev_test"
                fprintf(fid, 'split(dev/test): dev=1:%d, warmup=1:%d (not scored), tune=%d:%d, test=%d:%d\n', ...
                    dev_end, warmup_end, tune_start, tune_end, split.test_start, split.test_end);
                fprintf(fid, 'ratios(approx): warmup=%.2f, tune=%.2f, test=%.2f\n', split.train_ratio, split.val_ratio, split.test_ratio);
            else
                fprintf(fid, 'split: train=1:%d, val=%d:%d, test=%d:%d (ratios %.2f/%.2f/%.2f)\n', ...
                    split.train_end, split.val_start, split.val_end, split.test_start, split.test_end, ...
                    split.train_ratio, split.val_ratio, split.test_ratio);
            end
        fprintf(fid, 'fixed: tran_cost=%.6f, win_size=%d, epsilon=%.1f\n', opts.tran_cost, opts.win_size, opts.epsilon);
        fprintf(fid, 'update_mix=%.10g, max_turnover=%s, Q_clip_max=%s\n', ...
                best.update_mix, num2str(best.max_turnover, '%.10g'), num2str(best.Q_clip_max, '%.10g'));
        fprintf(fid, 'state_adaptive_trading=%d, risk_high_floor=%.10g\n', ...
            double(opts.state_adaptive_trading), double(opts.risk_high_floor));
        fprintf(fid, 'gating_mode=%s, trend_win=%d, trend_gamma=%.10g, risk_sigma_factor=%.10g\n', ...
            char(gating_mode), floor(double(opts.trend_win)), double(opts.trend_gamma), double(opts.risk_sigma_factor));
        fprintf(fid, 'val_objective=%s\n', best.val_objective);
        if val_objective == "log_wealth_turnover"
            fprintf(fid, 'turnover_penalty_lambda=%.10g\n', opts.turnover_penalty_lambda);
        end
            fprintf(fid, 'val_score=%.10f\n', best.val_score);
            if isfield(best, 'primary_log_wealth')
                fprintf(fid, 'primary_log_wealth=%.10f\n', best.primary_log_wealth);
            end
            if isfield(best, 'val_calmar')
                fprintf(fid, 'val_calmar=%.10f\n', best.val_calmar);
            end
            if isfield(best, 'val_turnover_mean')
                fprintf(fid, 'val_turnover_mean=%.10f\n', best.val_turnover_mean);
            end
            if isfield(best, 'val_turnover_threshold')
                fprintf(fid, 'val_turnover_threshold=%.10f\n', best.val_turnover_threshold);
            end
            fprintf(fid, 'val_wealth_folds=%s\n', mat2str(best.val_wealth_folds', 8));
            fprintf(fid, 'val_geom=%.10f\n', best.val_geom);
            fprintf(fid, 'test_wealth=%.10f\n', test_wealth);
            fprintf(fid, 'best params: L_percentile=%.1f, weight_inspect_wins=%d, risk_inspect_wins=%d, q_value=%.2f, reverse=%d, risk=%d\n', ...
                best.L_percentile, best.weight_inspect_wins, best.risk_inspect_wins, best.q_value, best.reverse_factor, best.risk_factor);
            fclose(fid);
        end

        fprintf('Best: val_score=%.6f, val_geom=%.6f, test_wealth=%.6f\n', best.val_score, best.val_geom, test_wealth);

        entry = struct();
        entry.dataset = dataset;
        entry.T = T;
        entry.N = N;
        entry.train_end = split.train_end;
        entry.val_start = split.val_start;
        entry.val_end = split.val_end;
        entry.test_start = split.test_start;
        entry.test_end = split.test_end;
        entry.train_ratio = split.train_ratio;
        entry.val_ratio = split.val_ratio;
        entry.test_ratio = split.test_ratio;
        entry.split_mode = char(split_mode);
        entry.dev_ratio = opts.dev_ratio;
        entry.dev_end = dev_end;
        entry.warmup_end = warmup_end;
        entry.tune_start = tune_start;
        entry.tune_end = tune_end;
        entry.K = K;
        entry.tran_cost = opts.tran_cost;
        entry.win_size = opts.win_size;
        entry.epsilon = opts.epsilon;
        entry.sharpe_annualization = opts.sharpe_annualization;
        entry.update_mix = best.update_mix;
        entry.max_turnover = best.max_turnover;
        entry.Q_clip_max = best.Q_clip_max;
        entry.state_adaptive_trading = double(opts.state_adaptive_trading);
        entry.risk_high_floor = double(opts.risk_high_floor);
        entry.gating_mode = char(gating_mode);
        entry.trend_win = floor(double(opts.trend_win));
        entry.trend_gamma = double(opts.trend_gamma);
        entry.risk_sigma_factor = double(opts.risk_sigma_factor);
        entry.val_objective = best.val_objective;
        entry.val_score = best.val_score;
        entry.val_geom = best.val_geom;
        entry.test_wealth = test_wealth;
        if isfield(best, 'primary_log_wealth')
            entry.primary_log_wealth = best.primary_log_wealth;
        else
            entry.primary_log_wealth = NaN;
        end
        if isfield(best, 'val_calmar')
            entry.val_calmar = best.val_calmar;
        else
            entry.val_calmar = NaN;
        end
        if isfield(best, 'val_turnover_mean')
            entry.val_turnover_mean = best.val_turnover_mean;
        else
            entry.val_turnover_mean = NaN;
        end
        if isfield(best, 'val_turnover_threshold')
            entry.val_turnover_threshold = best.val_turnover_threshold;
        else
            entry.val_turnover_threshold = NaN;
        end
        if isfield(best, 'val_sharpe')
            entry.val_sharpe = best.val_sharpe;
        else
            entry.val_sharpe = NaN;
        end
        entry.weight_inspect_wins = best.weight_inspect_wins;
        entry.risk_inspect_wins = best.risk_inspect_wins;
        entry.L_percentile = best.L_percentile;
        entry.q_value = best.q_value;
        entry.reverse_factor = best.reverse_factor;
        entry.risk_factor = best.risk_factor;
        entry.extreme_confirm_days = best.extreme_confirm_days;
        entry.high_confirm_days = best.high_confirm_days;
        summary(end + 1) = entry;
    end

    Tsum = struct2table(summary);
    clip_tag = "";
    if ~isempty(opts.Q_clip_max_values)
        clip_tag = "_QclipGrid";
    elseif ~isinf(opts.Q_clip_max)
        clip_tag = sprintf('_Qclip%.4g', opts.Q_clip_max);
    end
    if val_objective == "log_wealth_turnover"
        prefix = sprintf('ipt_fixed_%s_lambda%.4g%s', char(val_objective), opts.turnover_penalty_lambda, clip_tag);
    else
        prefix = sprintf('ipt_fixed_%s%s', char(val_objective), clip_tag);
    end
    if lower(string(opts.split_mode)) == "dev_test"
        tag = sprintf('dev%.0f_test%.0f', 100 * opts.dev_ratio, 100 * (1 - opts.dev_ratio));
    else
        tag = sprintf('%.0f_%.0f_%.0f', 100 * opts.train_ratio, 100 * opts.val_ratio, 100 * (1 - opts.train_ratio - opts.val_ratio));
    end
    run_tag = string(opts.run_tag);
    if strlength(run_tag) > 0
        run_tag = "_" + run_tag;
    end
    csv_path = fullfile(out_dir, prefix + "_summary_" + tag + "_" + grid_profile + run_tag + ".csv");
    writetable(Tsum, csv_path);

    txt_path = fullfile(out_dir, prefix + "_summary_" + tag + "_" + grid_profile + run_tag + ".txt");
    fid = fopen(txt_path, 'w');
    if fid ~= -1
        fprintf(fid, '%s\n', strjoin(Tsum.Properties.VariableNames, '\t'));
        for i = 1:height(Tsum)
            row = Tsum(i, :);
            parts = cell(1, width(Tsum));
            for j = 1:width(Tsum)
                v = row{1, j};
                if isnumeric(v)
                    parts{j} = num2str(v, '%.10g');
                else
                    parts{j} = string(v);
                end
            end
            fprintf(fid, '%s\n', strjoin(string(parts), '\t'));
        end
        fclose(fid);
    end

    fprintf('\nSaved: %s\nSaved: %s\n', csv_path, txt_path);
end

function [score, score_calmar, turnover_mean, score_sharpe] = eval_combo(idx, combos, tie_factors, weight_list, risk_list, L_percentiles, q_values, factor_values, update_mix_list, max_turnover_list, q_clip_list, extreme_confirm_days_list, high_confirm_days_list, near_risk_mode, risk_threshold_mode, near_L_high_cache, near_L_ext_cache, ...
    yar_weights_long_cache, yar_weights_near_cache, yar_ubah_long_cache, yar_ubah_near_cache, ...
    data, p_close, fold_ranges, tran_cost, win_size, epsilon, L_smoothing_alpha, objective, turnover_penalty_lambda, val_log_wealth_cap, val_sharpe_weight, default_update_mix, ...
    gating_mode, trend_win, trend_gamma, risk_sigma_factor, trend_guard_reversal, Q_clip_highrisk, state_adaptive_trading, risk_high_floor, adaptive_inertia_q, sharpe_annualization)

    wi = combos(idx, 1);
    ri = combos(idx, 2);
    li = combos(idx, 3);
    qi = combos(idx, 4);
    if tie_factors
        umi = combos(idx, 6);
        ti = combos(idx, 7);
        ci = combos(idx, 8);
        edi = combos(idx, 9);
        hdi = combos(idx, 10);
    else
        umi = combos(idx, 7);
        ti = combos(idx, 8);
        ci = combos(idx, 9);
        edi = combos(idx, 10);
        hdi = combos(idx, 11);
    end
    max_turnover = max_turnover_list(ti);
    q_clip_max = q_clip_list(ci);
    extreme_confirm_days = extreme_confirm_days_list(edi);
    high_confirm_days = high_confirm_days_list(hdi);

    weight_inspect_wins = weight_list(wi);
    risk_inspect_wins = risk_list(ri);
    q_value = q_values(qi);
    near_risk_halfwin = floor(risk_inspect_wins / 2);

    if tie_factors
        fi = combos(idx, 5);
        reverse_factor = factor_values(fi);
        risk_factor = reverse_factor;
    else
        revi = combos(idx, 5);
        rfi = combos(idx, 6);
        reverse_factor = factor_values(revi);
        risk_factor = factor_values(rfi);
    end
    if isempty(update_mix_list)
        update_mix = default_update_mix;
    else
        update_mix = update_mix_list(umi);
    end

    yar_weights_long = yar_weights_long_cache{wi};
    yar_weights_near = yar_weights_near_cache{wi};
    yar_ubah_long = yar_ubah_long_cache{wi, ri};
    yar_ubah_near = yar_ubah_near_cache{wi, ri};

    L_long_raw = compute_yar_percentile(yar_ubah_long(:, 1), L_percentiles(li));
    L_long_history = ipt_smooth_series(L_long_raw, L_smoothing_alpha);
    L_near_raw = compute_yar_percentile(yar_ubah_near(:, 1), L_percentiles(li));
    L_near_history = ipt_smooth_series(L_near_raw, L_smoothing_alpha);

    near_L_high = [];
    near_L_ext = [];
    risk_threshold_mode = lower(string(risk_threshold_mode));
    if risk_threshold_mode ~= "scale"
        near_L_high = near_L_high_cache{ri, qi};
        near_L_ext = near_L_ext_cache{ri, qi};
    end

    % active_function.m currently exposes the original hard gating signature only.
    [w_YAR, Q_factor] = active_function( ...
        yar_weights_long, yar_weights_near, ...
        yar_ubah_long, yar_ubah_near, ...
        data, weight_inspect_wins, ...
        reverse_factor, risk_factor, q_value, L_long_history, L_near_history);
    state_meta = [];
    Q_factor = clip_q(Q_factor, q_clip_max);

    K = size(fold_ranges, 1);
    fold_wealths = zeros(K, 1);
    fold_mdds = zeros(K, 1);
    fold_turnovers = zeros(K, 1);
    fold_sharpes = nan(K, 1);
    for k = 1:K
        update_mix_used = update_mix;
        max_turnover_used = max_turnover;
        if state_adaptive_trading && ~isempty(state_meta) && isfield(state_meta, 'g_weights')
            gw = state_meta.g_weights;
            risk_high = sum(gw(:, 4:5), 2);
            risk_high(~isfinite(risk_high)) = 0;
            update_mix_used = 1 - (1 - update_mix) .* risk_high;
            update_mix_used = max(1e-6, min(1, update_mix_used));
            if isinf(max_turnover)
                max_turnover_used = Inf;
            else
                denom = max(double(risk_high_floor), risk_high);
                max_turnover_used = max_turnover ./ denom;
                max_turnover_used(~isfinite(max_turnover_used)) = max_turnover;
            end
        end
        [fold_wealths(k), fold_mdds(k), fold_turnovers(k), fold_sharpes(k)] = eval_ipt_segment(data, p_close, w_YAR, Q_factor, ...
            win_size, tran_cost, epsilon, ...
            fold_ranges(k, 1), fold_ranges(k, 2), update_mix_used, max_turnover_used, sharpe_annualization, adaptive_inertia_q);
    end

    turnover_mean = mean(fold_turnovers);
    score_calmar = score_folds(fold_wealths, fold_mdds, fold_turnovers, "calmar", turnover_penalty_lambda, val_log_wealth_cap);

    score_sharpe = mean(fold_sharpes, 'omitnan');
    if ~isfinite(score_sharpe)
        score_sharpe = -Inf;
    end

    objective = lower(string(objective));
    if objective == "log_wealth_plus_sharpe"
        base_log = score_folds(fold_wealths, fold_mdds, fold_turnovers, "log_wealth", turnover_penalty_lambda, val_log_wealth_cap);
        score = base_log + double(val_sharpe_weight) * score_sharpe;
    else
        score = score_folds(fold_wealths, fold_mdds, fold_turnovers, objective, turnover_penalty_lambda, val_log_wealth_cap);
    end
end

function q = clip_q(q, q_clip_max)
    if isinf(q_clip_max)
        return;
    end
    q(q > q_clip_max) = q_clip_max;
    q(q < -q_clip_max) = -q_clip_max;
end

function y = quantile_base(x, q)
    % q in (0,1]
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        y = Inf;
        return;
    end
    x = sort(x, 'ascend');
    n = numel(x);
    if n == 1
        y = x(1);
        return;
    end
    q = min(max(double(q), 0), 1);
    if q == 1
        y = x(end);
        return;
    end
    pos = 1 + (n - 1) * q;
    lo = floor(pos);
    hi = ceil(pos);
    if lo == hi
        y = x(lo);
        return;
    end
    w = pos - lo;
    y = (1 - w) * x(lo) + w * x(hi);
end

function mode_out = ternary_prc_mode(mode_in)
    mode_in = lower(string(mode_in));
    if mode_in == "scale"
        mode_out = "scale";
    else
        mode_out = "near_prc";
    end
end

function [wealth, max_drawdown, turnover_mean, sharpe] = eval_ipt_segment(data, p_close, w_YAR, Q_factor, win_size, tran_cost, epsilon, start_idx, end_idx, update_mix, max_turnover, sharpe_annualization, adaptive_inertia_q)
    [T, N] = size(data);
    if end_idx > T
        error('end_idx out of range');
    end
    if nargin < 10 || isempty(update_mix)
        update_mix = 1;
    end
    if ~(isnumeric(update_mix) && (isscalar(update_mix) || numel(update_mix) == T))
        error('update_mix must be a scalar or a length-T vector.');
    end
    if any(update_mix(:) <= 0) || any(update_mix(:) > 1) || any(~isfinite(update_mix(:)))
        error('update_mix must be finite and in (0,1].');
    end
    if nargin < 11 || isempty(max_turnover)
        max_turnover = Inf;
    end
    if nargin < 13 || isempty(adaptive_inertia_q)
        adaptive_inertia_q = false;
    end
    if ~(isnumeric(max_turnover) && (isscalar(max_turnover) || numel(max_turnover) == T))
        error('max_turnover must be a scalar or a length-T vector.');
    end
    if isscalar(max_turnover)
        if ~(max_turnover > 0)
            error('max_turnover must be a positive scalar (use Inf to disable).');
        end
    else
        if any(max_turnover(:) <= 0) || any(~isfinite(max_turnover(:)))
            error('max_turnover vector must be finite and > 0 (use a scalar Inf to disable).');
        end
    end
    b_current = ones(N, 1) / N;
    b_prev = zeros(N, 1);
    wealth = 1;
    peak = 1;
    max_drawdown = 0;
    turnover_sum = 0;
    turnover_cnt = 0;

    r_cnt = 0;
    r_mean = 0;
    r_M2 = 0;

    for t = 1:end_idx
        turnover_t = sum(abs(b_current - b_prev));
        daily_incre = (data(t, :) * b_current) * (1 - tran_cost / 2 * turnover_t);
        if t >= start_idx
            wealth = wealth * daily_incre;
            turnover_sum = turnover_sum + turnover_t;
            turnover_cnt = turnover_cnt + 1;

            rr = double(daily_incre) - 1;
            if isfinite(rr)
                r_cnt = r_cnt + 1;
                d = rr - r_mean;
                r_mean = r_mean + d / r_cnt;
                r_M2 = r_M2 + d * (rr - r_mean);
            end

            if wealth > peak
                peak = wealth;
            else
                dd = 1 - wealth / peak;
                if dd > max_drawdown
                    max_drawdown = dd;
                end
            end
        end

        b_prev = b_current .* data(t, :)' / (data(t, :) * b_current);
        if t < end_idx
            b_next_raw = IPT(p_close, data, t, b_current, win_size, w_YAR, Q_factor);
            delta = b_next_raw - b_current;
            if isscalar(update_mix)
                alpha = update_mix;
            else
                alpha = update_mix(t);
            end
            if adaptive_inertia_q && numel(Q_factor) >= t && isfinite(Q_factor(t))
                alpha = alpha * (1 / (1 + abs(Q_factor(t))));
            end
            if isscalar(max_turnover)
                cap = max_turnover;
            else
                cap = max_turnover(t);
            end
            if ~isinf(cap)
                delta_turnover = sum(abs(delta));
                if delta_turnover > 0
                    alpha = min(alpha, cap / delta_turnover);
                else
                    alpha = 0;
                end
            end
            b_current = b_current + alpha * delta;
        end
    end

    if turnover_cnt > 0
        turnover_mean = turnover_sum / turnover_cnt;
    else
        turnover_mean = NaN;
    end

    if r_cnt >= 2
        r_std = sqrt(max(r_M2 / (r_cnt - 1), 0));
        if r_std > 0
            sharpe = sqrt(double(sharpe_annualization)) * (r_mean / r_std);
        else
            sharpe = NaN;
        end
    else
        sharpe = NaN;
    end
end

function wins_both = compute_wins_both_counts(scores_log, scores_sharpe, baseline_log, baseline_sharpe)
    baseline_log = double(baseline_log(:))';
    baseline_sharpe = double(baseline_sharpe(:))';
    if isempty(baseline_log) || isempty(baseline_sharpe)
        error('baseline_log/baseline_sharpe must be non-empty.');
    end
    if numel(baseline_log) ~= numel(baseline_sharpe)
        error('baseline_log and baseline_sharpe must have the same length.');
    end
    scores_log = double(scores_log(:));
    scores_sharpe = double(scores_sharpe(:));
    n = numel(scores_log);
    wins_both = -inf(n, 1);
    for i = 1:n
        if ~isfinite(scores_log(i)) || ~isfinite(scores_sharpe(i))
            wins_both(i) = -Inf;
            continue;
        end
        w = (scores_log(i) > baseline_log) & (scores_sharpe(i) > baseline_sharpe);
        wins_both(i) = sum(w);
    end
end

function [baseline_val_log, baseline_val_sharpe, baseline_names] = compute_baseline_val_scores(data, fold_ranges, tran_cost, win_size, epsilon, val_log_wealth_cap, sharpe_annualization)
    % Compute baseline validation metrics aligned with eval_combo:
    %   - baseline_val_log: mean(min(log(fold_terminal_wealth), cap)) across folds
    %   - baseline_val_sharpe: mean(fold_sharpe) across folds
    %
    % Runs each baseline once up to max(fold_end), then scores each fold by slicing
    % the daily return stream.

    baseline_names = lower(string({'ubah','bcrp','up','olmar2','rmr','anticor','corn','ppt','tppt'}));
    num_baselines = numel(baseline_names);
    K = size(fold_ranges, 1);
    max_end = max(fold_ranges(:, 2));
    if max_end < 2
        error('fold_ranges too short to evaluate baselines.');
    end
    data_run = data(1:max_end, :);
    tc = double(tran_cost);
    cap = double(val_log_wealth_cap);

    opts_env = struct();
    opts_env.quiet_mode = 1;
    opts_env.display_interval = 1000000;
    opts_env.progress = 0;
    opts_env.log_mode = 0;
    opts_env.mat_mode = 0;
    opts_env.analyze_mode = 0;
    opts_env.his = 0;

    % Baseline code lives outside this repo.
    base_dir = fileparts(fileparts(mfilename('fullpath'))); % .../matlab code
    olps_dir = 'H:/OLPS-master/Strategy';
    ppt_dir = fullfile(base_dir, 'PPT');
    tppt_dir = fullfile(base_dir, 'TPPT');

    % Keep a local log file for OLPS baselines that require fid.
    log_path = [tempname, '_olps_log.txt'];
    fid = fopen(log_path, 'w');
    if fid == -1
        error('Cannot open baseline log file: %s', log_path);
    end
    fid_cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    baseline_val_log = -inf(1, num_baselines);
    baseline_val_sharpe = -inf(1, num_baselines);

    % Add paths for OLPS, and run baselines one-by-one (avoid path conflicts with PPT/TPPT).
    addpath(olps_dir, '-begin');
    olps_cleanup = onCleanup(@() rmpath(olps_dir)); %#ok<NASGU>

    for bi = 1:num_baselines
        b = baseline_names(bi);
        try
            if b == "ubah"
                [~, ~, daily_ret] = ubah_run(fid, data_run, tc, opts_env);
            elseif b == "bcrp"
                [~, ~, daily_ret] = bcrp_run(fid, data_run, tc, opts_env);
            elseif b == "up"
                [~, ~, daily_ret] = up_run(fid, data_run, tc, opts_env);
            elseif b == "olmar2"
                [~, ~, daily_ret] = olmar2_run(fid, data_run, 10, 0.5, tc, opts_env);
            elseif b == "rmr"
                [~, ~, day_ret] = rmr_run(fid, data_run, 0.5, tc, 5, opts_env); %#ok<ASGLU>
                daily_ret = day_ret(:);
            elseif b == "anticor"
                [~, ~, daily_ret] = anticor_run(fid, data_run, 30, tc, opts_env);
            elseif b == "corn"
                [~, ~, daily_ret] = corn_run(fid, data_run, 5, 0.1, tc, opts_env);
            elseif b == "ppt"
                daily_ret = run_ppt_like_daily_ret(ppt_dir, data_run, win_size, epsilon, tc);
            elseif b == "tppt"
                daily_ret = run_ppt_like_daily_ret(tppt_dir, data_run, win_size, epsilon, tc);
            else
                error('Unknown baseline: %s', b);
            end

            daily_ret = double(daily_ret(:));
            if numel(daily_ret) < max_end
                error('baseline=%s returned daily_ret length=%d, expected >=%d', b, numel(daily_ret), max_end);
            end

            fold_logs = -inf(K, 1);
            fold_sharpes = -inf(K, 1);
            for k = 1:K
                s = fold_ranges(k, 1);
                e = fold_ranges(k, 2);
                [w, sh] = segment_wealth_sharpe(daily_ret, s, e, sharpe_annualization);
                lw = log(max(w, realmin));
                if isfinite(cap)
                    lw = min(lw, cap);
                end
                fold_logs(k) = lw;
                fold_sharpes(k) = sh;
            end

            baseline_val_log(bi) = mean(fold_logs, 'omitnan');
            baseline_val_sharpe(bi) = mean(fold_sharpes, 'omitnan');
            if ~isfinite(baseline_val_log(bi))
                baseline_val_log(bi) = -Inf;
            end
            if ~isfinite(baseline_val_sharpe(bi))
                baseline_val_sharpe(bi) = -Inf;
            end
        catch ME
            warning('Baseline precompute failed for %s: %s', b, ME.message);
            baseline_val_log(bi) = -Inf;
            baseline_val_sharpe(bi) = -Inf;
        end
    end
end

function daily_ret = run_ppt_like_daily_ret(model_dir, data, win_size, epsilon, tran_cost)
    addpath(model_dir, '-begin');
    cleanupObj = onCleanup(@() rmpath(model_dir)); %#ok<NASGU>
    clear PPT PPT_run simplex_projection_selfnorm2

    [T, N] = size(data);
    close_price = ones(T, N);
    for t = 2:T
        close_price(t, :) = close_price(t - 1, :) .* data(t, :);
    end

    daily_port = ones(N, 1) / N;
    daily_port_o = zeros(N, 1);
    daily_ret = ones(T, 1);
    for t = 1:T
        turnover_t = sum(abs(daily_port - daily_port_o));
        daily_ret(t, 1) = (data(t, :) * daily_port) * (1 - tran_cost / 2 * turnover_t);
        daily_port_o = daily_port .* data(t, :)' / (data(t, :) * daily_port);
        if t < T
            [daily_port_n, ~, ~] = PPT(close_price, data, t, daily_port, win_size, epsilon);
            daily_port = daily_port_n;
        end
    end
end

function [wealth, sharpe] = segment_wealth_sharpe(daily_ret, start_idx, end_idx, sharpe_annualization)
    if start_idx < 1 || end_idx > numel(daily_ret) || start_idx > end_idx
        error('Invalid segment indices.');
    end
    seg = double(daily_ret(start_idx:end_idx));
    if any(~isfinite(seg)) || any(seg <= 0)
        wealth = realmin;
        sharpe = -Inf;
        return;
    end
    wealth = prod(seg);
    rr = seg - 1;
    if numel(rr) >= 2
        s = std(rr, 0);
        if isfinite(s) && s > 0
            sharpe = sqrt(double(sharpe_annualization)) * (mean(rr) / s);
        else
            sharpe = -Inf;
        end
    else
        sharpe = -Inf;
    end
end

function [x_rel, info] = clip_xrel(x_rel, mode, fixed_bounds, prc_bounds)
    mode = lower(string(mode));
    x = x_rel(:);
    if any(~isfinite(x))
        error('x_rel contains non-finite values, cannot clip safely.');
    end

    if mode == "fixed"
        lo = fixed_bounds(1);
        hi = fixed_bounds(2);
    else
        lo = prctile(x, prc_bounds(1));
        hi = prctile(x, prc_bounds(2));
    end
    if ~(isfinite(lo) && isfinite(hi) && lo > 0 && hi > lo)
        error('Invalid clip bounds computed (lo=%.6g, hi=%.6g).', lo, hi);
    end

    below = x_rel < lo;
    above = x_rel > hi;
    x_rel(below) = lo;
    x_rel(above) = hi;

    info = struct();
    info.lo = lo;
    info.hi = hi;
    info.num_clipped = sum(below(:)) + sum(above(:));
end

function report_xrel_extremes(out_dir, dataset, x_rel, topk, run_tag)
    topk = max(0, floor(topk));
    if topk == 0
        return;
    end

    x = x_rel(:);
    if isempty(x)
        return;
    end

    [min_vals, min_idx] = mink(x, min(topk, numel(x)));
    [max_vals, max_idx] = maxk(x, min(topk, numel(x)));

    [T, N] = size(x_rel);
    [min_t, min_j] = ind2sub([T, N], min_idx);
    [max_t, max_j] = ind2sub([T, N], max_idx);

    kind = [repmat("min", numel(min_vals), 1); repmat("max", numel(max_vals), 1)];
    t_idx = [min_t(:); max_t(:)];
    asset_idx = [min_j(:); max_j(:)];
    val = [min_vals(:); max_vals(:)];

    Tout = table(kind, t_idx, asset_idx, val);

    tag = string(run_tag);
    if strlength(tag) > 0
        tag = "_" + tag;
    end
    out_csv = fullfile(out_dir, sprintf('xrel_extremes_%s%s.csv', dataset, tag));
    writetable(Tout, out_csv);
end

function score = score_folds(fold_wealths, fold_mdds, fold_turnovers, objective, turnover_penalty_lambda, val_log_wealth_cap)
    objective = lower(string(objective));
    cap = double(val_log_wealth_cap);
    switch objective
        case "log_wealth"
            log_ret = log(max(fold_wealths, realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            score = mean(log_ret);
        case "calmar"
            log_ret = log(max(fold_wealths, realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            denom = max(fold_mdds, 1e-12);
            score = mean(log_ret ./ denom);
        case "log_wealth_turnover"
            log_ret = log(max(fold_wealths, realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            score = mean(log_ret) - turnover_penalty_lambda * mean(fold_turnovers, 'omitnan');
        case "log_wealth_q25"
            log_ret = log(max(fold_wealths, realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            score = prctile(log_ret, 25);
        case "log_wealth_min"
            log_ret = log(max(fold_wealths, realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            score = min(log_ret);
        case "log_wealth_last"
            log_ret = log(max(fold_wealths(end), realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            score = log_ret;
        case "log_wealth_recent"
            log_ret = log(max(fold_wealths, realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            log_ret(~isfinite(log_ret)) = -Inf;
            w = (1:numel(log_ret))';
            w = w / sum(w);
            score = sum(w .* log_ret);
        case "log_wealth_stable"
            log_ret = log(max(fold_wealths, realmin));
            if isfinite(cap)
                log_ret = min(log_ret, cap);
            end
            log_ret(~isfinite(log_ret)) = -Inf;
            lambda = 0.5;
            score = mean(log_ret) - lambda * std(log_ret, 0);
        otherwise
            error('Unsupported objective: %s', objective);
    end
end

function r = rank_desc(x)
    % Rank in descending order: best (largest) gets rank 1.
    x = double(x(:));
    x(~isfinite(x)) = -Inf;
    [~, ord] = sort(x, 'descend');
    r = zeros(size(x));
    r(ord) = 1:numel(x);
end

