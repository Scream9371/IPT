function out_dir = run_core(varargin)
    %run_core One-click reproduction of IPT mainline results.

    script_dir = fileparts(mfilename('fullpath')); repo_root = fileparts(script_dir); addpath(script_dir);
    p = inputParser;
    addParameter(p, 'data_dir', fullfile(repo_root, 'Data Set'));
    addParameter(p, 'datasets', {'nyse-n', 'nyse-o', 'ndx', 'inv500', 'inv30', 'multi_asset'});
    addParameter(p, 'out_dir', fullfile(repo_root, 'results', ['ipt_mainline_', datestr(now, 'yyyymmdd_HHMMSS')]));
    parse(p, varargin{:}); opts = p.Results; out_dir = opts.out_dir;
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    cfg = struct('year', 252, 'win_len', 1260, 'train_len', 1008, 'test_len', 252, 'step', 252, ...
        'win_size', 5, 'tran_cost', 0.001, 'q', 0.3, 'L_pct', 99, 'reverse_factor', 10, 'risk_factor', 10, ...
        'w_inspect_wins', 252);

    rows = {};

    for di = 1:numel(opts.datasets)
        ds = opts.datasets{di}; mat_path = fullfile(opts.data_dir, [ds, '.mat']);
        if ~exist(mat_path, 'file'), fprintf('skip %s (missing %s)\n', ds, mat_path); continue; end
        data = load_dataset_matrix(mat_path); if isempty(data), fprintf('skip %s (invalid data)\n', ds); continue; end
        if size(data, 1) < size(data, 2), data = data'; end; [T, ~] = size(data);
        if T < cfg.win_len, fprintf('skip %s (T=%d < %d)\n', ds, T, cfg.win_len); continue; end

        cw_list = []; ann_list = []; sharpe_list = []; sortino_list = []; calmar_list = []; sterling_list = []; mdd_list = []; win_starts = []; daily_ret_blocks = {};

        for s = 1:cfg.step:(T - cfg.win_len + 1)
            x_rel = data(s:s + cfg.win_len - 1, :);
            w_full = compute_yar_weights(x_rel, cfg.w_inspect_wins); w_half = compute_yar_weights(x_rel, floor(cfg.w_inspect_wins / 2));
            idx_series = index_compute(x_rel);
            af_win = 21;
            af_full = compute_yar_index_factor(idx_series(cfg.w_inspect_wins - af_win + 1:end, :), af_win);
            af_half = compute_yar_index_factor(idx_series(floor(cfg.w_inspect_wins / 2) - floor(af_win / 2) + 1:end, :), floor(af_win / 2));
            L = adaptive_L_from_train(af_full, cfg.train_len, cfg.w_inspect_wins, cfg.L_pct); if isempty(L) || ~isfinite(L) || L <= 0, continue; end
            [w, r] = active_function(w_full, w_half, af_full, af_half, x_rel, cfg.w_inspect_wins, cfg.reverse_factor, cfg.risk_factor, L, cfg.q);
            r = apply_state_filter(r, idx_series, cfg.risk_factor);

            ts = cfg.train_len + 1; te = ts + cfg.test_len - 1; [cw_full, ~, ~] = IPT_run(x_rel, cfg.win_size, cfg.tran_cost, w, r);
            test_cw = cw_full(ts:te) / cw_full(ts - 1); daily_ret = [test_cw(1); test_cw(2:end) ./ test_cw(1:end - 1)];
            cw_list(end + 1, 1) = test_cw(end); win_starts(end + 1, 1) = s; daily_ret_blocks{end + 1, 1} = daily_ret; %#ok<AGROW>
            [ann_ret, sharpe, sortino, calmar, sterling, mdd] = compute_metrics(daily_ret, test_cw);
            ann_list(end + 1, 1) = ann_ret; sharpe_list(end + 1, 1) = sharpe; sortino_list(end + 1, 1) = sortino; calmar_list(end + 1, 1) = calmar; sterling_list(end + 1, 1) = sterling; mdd_list(end + 1, 1) = mdd; %#ok<AGROW>
        end

        if isempty(cw_list), fprintf('skip %s (no valid rolling windows)\n', ds); continue; end

        rows(end + 1, :) = {ds, mean(cw_list), mean(ann_list), mean(sharpe_list), mean(sortino_list), mean(calmar_list), mean(sterling_list), mean(mdd_list), numel(cw_list)}; %#ok<AGROW>
        save(fullfile(out_dir, sprintf('ipt-%s_roll5y.mat', ds)), 'cw_list', 'ann_list', 'sharpe_list', 'sortino_list', 'calmar_list', 'sterling_list', 'mdd_list', 'win_starts', 'daily_ret_blocks');
        fprintf('dataset %s: CW_mean=%.4f, windows=%d\n', ds, mean(cw_list), numel(cw_list));
    end

    write_summary_csv(fullfile(out_dir, 'rolling5y_summary.csv'), rows);
    fprintf('saved %s\n', out_dir);
end

function L = adaptive_L_from_train(af_full, train_len, w_inspect_wins, L_pct)
    L = []; train_max_i = train_len - w_inspect_wins;

    if train_max_i >= 1
        y = af_full(1:train_max_i); y = y(isfinite(y)); if ~isempty(y), L = prctile(y, L_pct); return; end
    end

    y = af_full(isfinite(af_full)); if ~isempty(y), L = prctile(y, L_pct); end
end

function write_summary_csv(path, rows)
    fid = fopen(path, 'w'); fprintf(fid, 'dataset,cw_mean,ann_ret_mean,sharpe_mean,sortino_mean,calmar_mean,sterling_mean,mdd_mean,n_windows\n');
    for i = 1:size(rows, 1), r = rows(i, :); fprintf(fid, '%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d\n', r{1}, r{2}, r{3}, r{4}, r{5}, r{6}, r{7}, r{8}, r{9}); end
    fclose(fid);
end

function data = load_dataset_matrix(mat_path)
    S = load(mat_path); data = [];
    if isfield(S, 'data'), data = S.data; return; end
    if isfield(S, 'msci'), data = S.msci; return; end
    fns = fieldnames(S); for i = 1:numel(fns), v = S.(fns{i}); if isnumeric(v) && ismatrix(v), data = v; return; end, end
end

function r = apply_state_filter(r, market_ret, risk_factor)
    k = 21;
    dd0 = 0.05;
    m0 = 0;
    hold_n = 3;
    T = numel(r); m_pre = nan(T, 1); dd_pre = nan(T, 1);

    for t = (k + 1):T
        seg = market_ret(t - k:t - 1); m_pre(t) = prod(seg) - 1; cum = cumprod(seg); peak = cum(1); dd = 0;
        for j = 1:numel(cum), if cum(j) > peak, peak = cum(j); end, dd = max(dd, (peak - cum(j)) / peak); end
        dd_pre(t) = dd;
    end

    for t = 1:T

        if r(t) == 2 * risk_factor
            ok = isfinite(m_pre(t)) && isfinite(dd_pre(t)) && (m_pre(t) < m0) && (dd_pre(t) > dd0);
            if ~ok, r(t) = 0; end
        end

    end

    if hold_n <= 1, return; end
    r2 = r;

    for t = 1:T
        if r(t) == 0, continue; end
        ok = true; for h = 0:(hold_n - 1), if t - h < 1 || r(t - h) == 0, ok = false; break; end, end
        if ~ok, r2(t) = 0; end
    end

    r = r2;
end

function [ann_ret, sharpe, sortino, calmar, sterling, mdd] = compute_metrics(daily_ret, cum_wealth)
    r = daily_ret - 1; if isempty(r), [ann_ret, sharpe, sortino, calmar, sterling, mdd] = deal(NaN); return; end
    ann_ret = prod(daily_ret) ^ (252 / numel(daily_ret)) - 1; s = std(r); sharpe = NaN; if s > 0, sharpe = mean(r) / s * sqrt(252); end
    dd = sqrt(mean(min(r, 0) .^ 2)); sortino = NaN; if dd > 0, sortino = mean(r) / dd * sqrt(252); end
    mdd = max_drawdown(cum_wealth); avg_dd = avg_drawdown(cum_wealth); calmar = NaN; sterling = NaN;
    if mdd > 0, calmar = ann_ret / mdd; end; if avg_dd > 0, sterling = ann_ret / avg_dd; end
end

function avg_dd = avg_drawdown(cum_wealth)
    peak = -inf; dd = zeros(numel(cum_wealth), 1);
    for t = 1:numel(cum_wealth), if cum_wealth(t) > peak, peak = cum_wealth(t); end, dd(t) = (peak - cum_wealth(t)) / max(peak, 1e-12); end
    avg_dd = mean(dd);
end

function mdd = max_drawdown(cum_wealth)
    peak = -inf; mdd = 0;
    for t = 1:numel(cum_wealth), if cum_wealth(t) > peak, peak = cum_wealth(t); end, dd = (peak - cum_wealth(t)) / max(peak, 1e-12); if dd > mdd, mdd = dd; end, end
end
