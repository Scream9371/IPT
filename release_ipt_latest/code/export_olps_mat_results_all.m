function export_olps_mat_results_all(varargin)
% export_olps_mat_results_all - Export IPT/PPT/TPPT results in OLPS .mat format.
%
% This script reproduces the variable layout of OLPS "Log/Mat/*.mat" files:
%   cum_ret, cumprod_ret, daily_ret, ra_ret, run_time, daily_portfolio
%
% It runs each model over the full history (so the test-period portfolio has
% access to past), but only exports the tail test segment (default tail40 via
% dev_ratio=0.6).
%
% Defaults:
%   split_mode: dev/test with dev_ratio=0.6 (tail40)
%   win_size=5, epsilon=100, tran_cost=0.001
%   IPT params are read from the best-per-dataset summary produced by
%   run_ipt_fixed_test (B strategy + Qclip grid).

    p = inputParser;
    addParameter(p, 'dev_ratio', 0.6);
    addParameter(p, 'win_size', 5);
    addParameter(p, 'epsilon', 100);
    addParameter(p, 'tran_cost', 0.001);
    addParameter(p, 'L_smoothing_alpha', 0.2);
    addParameter(p, 'ipt_summary_csv', fullfile('results_fixed_params', 'ipt_fixed_log_wealth_QclipGrid_summary_dev60_test40_robust_adaptTurn_capSearch_QclipGrid.csv'));
    addParameter(p, 'out_dir', 'results');
    addParameter(p, 'timestamp', datestr(now, 'yyyy-mmdd-HH-MM'));
    parse(p, varargin{:});
    opts = p.Results;

    script_dir = fileparts(mfilename('fullpath'));
    data_dir = fullfile(script_dir, 'Data Set');
    out_dir = fullfile(script_dir, opts.out_dir);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % Load per-dataset best IPT params.
    ipt_summary_path = fullfile(script_dir, opts.ipt_summary_csv);
    if ~exist(ipt_summary_path, 'file')
        error('IPT summary CSV not found: %s', ipt_summary_path);
    end
    ipt_tbl = readtable(ipt_summary_path);
    if ~ismember('dataset', ipt_tbl.Properties.VariableNames)
        error('IPT summary CSV missing column: dataset');
    end

    % OLPS analysis functions for ra_ret (14x1).
    addpath('H:/OLPS-master/Strategy', '-begin');

    files = dir(fullfile(data_dir, '*.mat'));
    if isempty(files)
        error('No datasets found in %s', data_dir);
    end
    [~, order] = sort({files.name});
    files = files(order);

    for i = 1:numel(files)
        dataset = erase(files(i).name, '.mat');
        data_path = fullfile(data_dir, files(i).name);
        S = load(data_path, 'data');
        data = S.data;
        [T, N] = size(data); %#ok<ASGLU>

        dev = ipt_dev_test_split(T, 'dev_ratio', opts.dev_ratio);
        test_start = dev.test_start;
        test_end = dev.test_end;
        test_idx = test_start:test_end;
        data_test = data(test_idx, :);

        fprintf('\n=== Export tail test: %s (T=%d, N=%d), test=%d:%d ===\n', dataset, T, N, test_start, test_end);

        % 1) PPT
        [cum_ret, cumprod_ret, daily_ret, ra_ret, run_time, daily_portfolio] = ...
            run_ppt_like('PPT', data, test_start, test_end, opts.win_size, opts.epsilon, opts.tran_cost);
        save(fullfile(out_dir, sprintf('ppt-%s_tail40-%s.mat', dataset, opts.timestamp)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'ra_ret', 'run_time', 'daily_portfolio');

        % 2) TPPT
        [cum_ret, cumprod_ret, daily_ret, ra_ret, run_time, daily_portfolio] = ...
            run_ppt_like('TPPT', data, test_start, test_end, opts.win_size, opts.epsilon, opts.tran_cost);
        save(fullfile(out_dir, sprintf('tppt-%s_tail40-%s.mat', dataset, opts.timestamp)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'ra_ret', 'run_time', 'daily_portfolio');

        % 3) IPT (params from summary)
        row = ipt_tbl(strcmpi(string(ipt_tbl.dataset), string(dataset)), :);
        if height(row) ~= 1
            error('Expected 1 row for dataset=%s in %s, got %d', dataset, ipt_summary_path, height(row));
        end

        ipt_params = struct();
        ipt_params.weight_inspect_wins = row.weight_inspect_wins(1);
        ipt_params.risk_inspect_wins = row.risk_inspect_wins(1);
        ipt_params.L_percentile = row.L_percentile(1);
        ipt_params.q_value = row.q_value(1);
        ipt_params.reverse_factor = row.reverse_factor(1);
        ipt_params.risk_factor = row.risk_factor(1);
        ipt_params.Q_clip_max = row.Q_clip_max(1);
        ipt_params.max_turnover = row.max_turnover(1);

        [cum_ret, cumprod_ret, daily_ret, ra_ret, run_time, daily_portfolio] = ...
            run_ipt_with_params(data, data_test, test_start, test_end, opts.win_size, opts.epsilon, opts.tran_cost, opts.L_smoothing_alpha, ipt_params);
        save(fullfile(out_dir, sprintf('ipt-%s_tail40-%s.mat', dataset, opts.timestamp)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'ra_ret', 'run_time', 'daily_portfolio');
    end

    rmpath('H:/OLPS-master/Strategy');
end

function [cum_ret, cumprod_ret, daily_ret, ra_ret, run_time, daily_portfolio] = run_ppt_like(model_kind, data, test_start, test_end, win_size, epsilon, tran_cost)
    base_dir = fileparts(fileparts(mfilename('fullpath')));
    model_dir = fullfile(base_dir, model_kind);
    addpath(model_dir, '-begin');
    clear PPT PPT_run simplex_projection_selfnorm2

    start_watch = tic;

    [T, N] = size(data);
    close_price = ones(T, N);
    for t = 2:T
        close_price(t, :) = close_price(t - 1, :) .* data(t, :);
    end

    daily_port = ones(N, 1) / N;
    daily_port_o = zeros(N, 1);
    daily_ret_full = ones(test_end, 1);
    port_full = zeros(test_end, N);

    for t = 1:test_end
        port_full(t, :) = daily_port';
        turnover_t = sum(abs(daily_port - daily_port_o));
        daily_ret_full(t, 1) = (data(t, :) * daily_port) * (1 - tran_cost / 2 * turnover_t);

        daily_port_o = daily_port .* data(t, :)' / (data(t, :) * daily_port);
        if t < test_end
            [daily_port_n, ~, ~] = PPT(close_price, data, t, daily_port, win_size, epsilon);
            daily_port = daily_port_n;
        end
    end

    run_time = toc(start_watch);

    daily_ret = daily_ret_full(test_start:test_end);
    cumprod_ret = cumprod(daily_ret);
    cum_ret = cumprod_ret(end);
    daily_portfolio = port_full(test_start:test_end, :);

    ra_ret = olps_ra_ret(data(test_start:test_end, :), cum_ret, cumprod_ret, daily_ret);

    rmpath(model_dir);
    clear PPT PPT_run simplex_projection_selfnorm2
end

function [cum_ret, cumprod_ret, daily_ret, ra_ret, run_time, daily_portfolio] = run_ipt_with_params(data, data_test, test_start, test_end, win_size, epsilon, tran_cost, L_smoothing_alpha, P)
    start_watch = tic;

    [T, ~] = size(data);

    p_close = ones(T, size(data, 2));
    for t = 2:T
        p_close(t, :) = p_close(t - 1, :) .* data(t, :);
    end

    ratio = ubah_price_ratio(data);

    w = P.weight_inspect_wins;
    r = P.risk_inspect_wins;
    half_weight = floor(w / 2);
    half_risk = floor(r / 2);

    start_long = w - r + 1;
    start_near = half_weight - half_risk + 1;
    if start_long < 1 || start_near < 1
        error('Invalid windows for IPT params (w=%d, r=%d).', w, r);
    end

    yar_weights_long = yar_weights(data, w);
    yar_weights_near = yar_weights(data, half_weight);
    yar_ubah_long = yar_ubah(ratio(start_long:T, :), r);
    yar_ubah_near = yar_ubah(ratio(start_near:T, :), half_risk);

    L_raw = compute_yar_percentile(yar_ubah_long(:, 1), P.L_percentile);
    L_history = ipt_smooth_series(L_raw, L_smoothing_alpha);

    [w_YAR, Q_factor] = active_function( ...
        yar_weights_long, yar_weights_near, ...
        yar_ubah_long, yar_ubah_near, ...
        data, w, ...
        P.reverse_factor, P.risk_factor, P.q_value, L_history);

    Q_factor = clip_q(Q_factor, P.Q_clip_max);

    [cum_full, daily_full, b_hist] = IPT_run(data, win_size, tran_cost, w_YAR, Q_factor, epsilon, 1, P.max_turnover);

    run_time = toc(start_watch);

    daily_ret = daily_full(test_start:test_end);
    cumprod_ret = cumprod(daily_ret);
    cum_ret = cumprod_ret(end);
    daily_portfolio = b_hist(:, test_start:test_end)'; % (n_test x N)

    ra_ret = olps_ra_ret(data_test, cum_ret, cumprod_ret, daily_ret);
end

function q = clip_q(q, q_clip_max)
    if isinf(q_clip_max)
        return;
    end
    q(q > q_clip_max) = q_clip_max;
    q(q < -q_clip_max) = -q_clip_max;
end

function ra_ret = olps_ra_ret(data_slice, cum_ret, cumprod_ret, daily_ret)
    opts = struct();
    opts.quiet_mode = 1;
    opts.display_interval = 500;
    opts.log_mode = 0;
    opts.mat_mode = 0;
    opts.analyze_mode = 1;
    ra_ret = ra_result_analyze(1, data_slice, cum_ret, cumprod_ret, daily_ret, opts);
end

