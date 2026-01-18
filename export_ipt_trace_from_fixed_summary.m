function export_ipt_trace_from_fixed_summary(dataset, varargin)
% export_ipt_trace_from_fixed_summary - Export IPT trace using fixed win_size/epsilon and fixed-summary params.
%
% This reads the best IPT-specific parameters from the fixed-parameter summary produced by
% ipt_fixed_test (e.g., ipt_fixed_log_wealth_summary.csv), then replays IPT to export:
%   - ipt_trace_<dataset>.csv (full timeline with segment labels)
%   - figs_<dataset>_<segment>/ (paper-friendly plots, optional)
%
% It does NOT perform any grid search.

    p = inputParser;
    addParameter(p, 'summary_csv', fullfile(pwd, 'Investment-potential-tracking', 'results_fixed_params', 'ipt_fixed_log_wealth_summary.csv'));
    addParameter(p, 'data_dir', fullfile(pwd, 'Investment-potential-tracking', 'Data Set'));
    addParameter(p, 'out_dir', fullfile(pwd, 'Investment-potential-tracking', 'results_fixed_params'));
    addParameter(p, 'L_smoothing_alpha', 0.2);
    addParameter(p, 'make_plots', true);
    addParameter(p, 'plot_segment', 'test'); % 'train' | 'val' | 'test' | 'all'
    parse(p, varargin{:});
    opts = p.Results;

    if ~isfile(opts.summary_csv)
        error('Missing summary CSV: %s', opts.summary_csv);
    end

    S = readtable(opts.summary_csv);
    dataset = string(dataset);
    row_idx = find(string(S.dataset) == dataset, 1);
    if isempty(row_idx)
        error('Dataset not found in summary: %s', dataset);
    end

    data_path = fullfile(opts.data_dir, dataset + ".mat");
    if ~isfile(data_path)
        error('Missing dataset file: %s', data_path);
    end

    D = load(data_path, 'data');
    data = D.data;
    [T, N] = size(data);

    win_size = S.win_size(row_idx);
    epsilon = S.epsilon(row_idx);
    tran_cost = S.tran_cost(row_idx);
    weight_inspect_wins = S.weight_inspect_wins(row_idx);
    risk_inspect_wins = S.risk_inspect_wins(row_idx);
    L_percentile = S.L_percentile(row_idx);
    q_value = S.q_value(row_idx);
    reverse_factor = S.reverse_factor(row_idx);
    risk_factor = S.risk_factor(row_idx);

    split = struct();
    split.train_end = S.train_end(row_idx);
    split.val_start = S.val_start(row_idx);
    split.val_end = S.val_end(row_idx);
    split.test_start = S.test_start(row_idx);
    split.test_end = S.test_end(row_idx);

    ratio = ubah_price_ratio(data);

    yar_weights_long = yar_weights(data, weight_inspect_wins);
    yar_weights_near = yar_weights(data, floor(weight_inspect_wins / 2));

    start_long = weight_inspect_wins - risk_inspect_wins + 1;
    half_weight = floor(weight_inspect_wins / 2);
    half_risk = floor(risk_inspect_wins / 2);
    start_near = half_weight - half_risk + 1;

    yar_ubah_long = yar_ubah(ratio(start_long:T, :), risk_inspect_wins);
    yar_ubah_near = yar_ubah(ratio(start_near:T, :), half_risk);

    L_raw = compute_yar_percentile(yar_ubah_long(:, 1), L_percentile);
    L_history = ipt_smooth_series(L_raw, opts.L_smoothing_alpha);

    [w_YAR, Q_factor, state_meta] = active_function( ...
        yar_weights_long, yar_weights_near, ...
        yar_ubah_long, yar_ubah_near, ...
        data, weight_inspect_wins, ...
        reverse_factor, risk_factor, q_value, L_history);

    [wealth_global, ~, b_history] = IPT_run(data, win_size, tran_cost, w_YAR, Q_factor, epsilon);
    weights = b_history';

    daily_incre = zeros(T, 1);
    daily_incre(1) = wealth_global(1);
    for t = 2:T
        if wealth_global(t - 1) == 0
            daily_incre(t) = 0;
        else
            daily_incre(t) = wealth_global(t) / wealth_global(t - 1);
        end
    end

    turnover = nan(T, 1);
    for t = 2:T
        turnover(t) = sum(abs(weights(t, :) - weights(t - 1, :)));
    end

    yar_exposure = nan(T, 1);
    yar_corr = nan(T, 1);
    for t = 1:T
        w = w_YAR(t, :)';
        b = weights(t, :)';
        yar_exposure(t) = b' * w;
        yar_corr(t) = ipt_safe_corr(b, w);
    end

    segment = repmat("train", T, 1);
    segment(split.val_start:split.val_end) = "val";
    segment(split.test_start:split.test_end) = "test";

    state_code = state_meta.state_code(:);
    state = arrayfun(@ipt_state_name, state_code);

    [wealth_seg, wealth_before] = ipt_compute_segment_wealth(daily_incre, segment);

    if ~exist(opts.out_dir, 'dir')
        mkdir(opts.out_dir);
    end

    trace_csv = fullfile(opts.out_dir, sprintf('ipt_trace_%s.csv', dataset));
    weight_names = arrayfun(@(i) sprintf('w_%d', i), 1:N, 'UniformOutput', false);
    base_tbl = table((1:T)', segment, wealth_seg, wealth_before, wealth_global, daily_incre, turnover, Q_factor(:), ...
        state_code, state, state_meta.L(:), state_meta.yar_ubah_long(:), state_meta.yar_ubah_near(:), yar_exposure, yar_corr, ...
        'VariableNames', {'t','segment','wealth','wealth_before','wealth_global','daily_incre','turnover','Q','state_code','state','L','yar_ubah_long','yar_ubah_near','yar_exposure','yar_corr'});
    weight_tbl = array2table(weights, 'VariableNames', weight_names);
    writetable([base_tbl, weight_tbl], trace_csv);

    if opts.make_plots
        figs_dir = fullfile(opts.out_dir, sprintf('figs_%s_%s', dataset, string(opts.plot_segment)));
        plot_ipt_three_state_trace(trace_csv, figs_dir, 'segment_filter', opts.plot_segment);
    end
end
