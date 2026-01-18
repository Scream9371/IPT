function run_ablation_export(varargin)
    % RUN_ABLATION_EXPORT Export IPT variants for ablation study
    % Based on export_fixed_ipt_results.m but with overrides for ablation.

    p = inputParser;
    addParameter(p, 'summary_csv', '');
    addParameter(p, 'algo_name', 'ipt_variant');
    addParameter(p, 'data_dir', fullfile(fileparts(mfilename('fullpath')), 'Data Set'));
    addParameter(p, 'results_dir', fullfile(fileparts(mfilename('fullpath')), 'results'));
    addParameter(p, 'L_smoothing_alpha', 0.2);
    addParameter(p, 'Q_smoothing_alpha', 0);
    addParameter(p, 'adaptive_inertia_q', false);
    addParameter(p, 'near_risk_mode', 'by_weight');

    % Ablation overrides
    addParameter(p, 'force_no_inertia', false);
    addParameter(p, 'force_no_qclip', false);
    addParameter(p, 'force_zero_cost', false);
    addParameter(p, 'force_no_orth', false);

    parse(p, varargin{:});
    opts = p.Results;

    if isempty(opts.summary_csv) || ~isfile(opts.summary_csv)
        error('summary_csv is required and must exist.');
    end

    if ~exist(opts.results_dir, 'dir')
        mkdir(opts.results_dir);
    end

    Tsum = readtable(opts.summary_csv);

    if ~any(strcmp(Tsum.Properties.VariableNames, 'dataset'))
        error('summary_csv missing column: dataset');
    end

    algo_name = char(string(opts.algo_name));

    for i = 1:height(Tsum)
        dataset = char(string(Tsum.dataset(i)));
        data_path = fullfile(opts.data_dir, dataset + ".mat");

        if ~isfile(data_path)
            warning('Dataset not found: %s (skipped)', data_path);
            continue;
        end

        S = load(data_path, 'data');
        data = S.data;
        [T, N] = size(data);

        p_close = ones(T, N);

        for t = 2:T
            p_close(t, :) = p_close(t - 1, :) .* data(t, :);
        end

        % Params from summary
        tran_cost = double(Tsum.tran_cost(i));

        if opts.force_zero_cost
            tran_cost = 0;
        end

        win_size = double(Tsum.win_size(i));

        if any(strcmp(Tsum.Properties.VariableNames, 'epsilon'))
            epsilon = double(Tsum.epsilon(i));
        else
            epsilon = 100;
        end

        update_mix = double(Tsum.update_mix(i));

        if opts.force_no_inertia
            update_mix = 1;
        end

        max_turnover = double(Tsum.max_turnover(i));

        Q_clip_max = double(Tsum.Q_clip_max(i));

        if opts.force_no_qclip
            Q_clip_max = Inf;
        end

        weight_inspect_wins = double(Tsum.weight_inspect_wins(i));
        risk_inspect_wins = double(Tsum.risk_inspect_wins(i));
        L_percentile = double(Tsum.L_percentile(i));
        q_value = double(Tsum.q_value(i));

        if any(strcmp(Tsum.Properties.VariableNames, 'reverse_factor'))
            reverse_factor = double(Tsum.reverse_factor(i));
        elseif any(strcmp(Tsum.Properties.VariableNames, 'risk_factor'))
            reverse_factor = double(Tsum.risk_factor(i));
        else
            error('summary_csv missing column: reverse_factor or risk_factor');
        end

        if any(strcmp(Tsum.Properties.VariableNames, 'risk_factor'))
            risk_factor = double(Tsum.risk_factor(i));
        else
            risk_factor = reverse_factor;
        end

        near_risk_mode = lower(string(opts.near_risk_mode));

        test_start = double(Tsum.test_start(i));
        test_end = double(Tsum.test_end(i));

        ratio = ubah_price_ratio(data);
        half_weight = floor(weight_inspect_wins / 2);
        half_risk = floor(risk_inspect_wins / 2);
        start_long = weight_inspect_wins - risk_inspect_wins + 1;
        start_near = half_weight - half_risk + 1;

        yar_weights_long = yar_weights(data, weight_inspect_wins);
        yar_weights_near = yar_weights(data, half_weight);
        yar_ubah_long = yar_ubah(ratio(start_long:T, :), risk_inspect_wins);

        if near_risk_mode == "by_weight"
            yar_ubah_near = yar_ubah(ratio(start_near:T, :), half_risk);
        else
            yar_ubah_near = yar_ubah(ratio, half_risk);
        end

        L_raw = compute_yar_percentile(yar_ubah_long(:, 1), L_percentile);
        L_history = ipt_smooth_series(L_raw, opts.L_smoothing_alpha);

        [w_YAR, Q_factor] = active_function( ...
            yar_weights_long, yar_weights_near, ...
            yar_ubah_long, yar_ubah_near, ...
            data, weight_inspect_wins, ...
            risk_factor, q_value, L_history);

        if opts.Q_smoothing_alpha > 0
            Q_factor = ipt_smooth_series(Q_factor, opts.Q_smoothing_alpha);
        end

        Q_factor = clip_q_local(Q_factor, Q_clip_max);

        [cum_wealth, daily_incre_fact, b_history] = ipt_run_with_inertia( ...
            p_close, data, win_size, epsilon, tran_cost, w_YAR, Q_factor, update_mix, max_turnover, opts.adaptive_inertia_q, opts.force_no_orth);

        out_path = fullfile(opts.results_dir, sprintf('%s-%s_tail40.mat', algo_name, dataset));

        idx0 = round(test_start);
        idx1 = round(test_end);
        daily_ret = daily_incre_fact(idx0:idx1);
        cumprod_ret = cumprod(daily_ret);
        daily_portfolio = b_history(:, idx0:idx1);
        save(out_path, 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        fprintf('Saved: %s\n', out_path);
    end

end

function q = clip_q_local(q, q_clip_max)

    if isinf(q_clip_max)
        return;
    end

    q(q > q_clip_max) = q_clip_max;
    q(q < -q_clip_max) = -q_clip_max;
end

function [cum_wealth, daily_incre_fact, b_history] = ipt_run_with_inertia(p_close, x_rel, win_size, epsilon, tran_cost, w_YAR, Q_factor, update_mix, max_turnover, adaptive_inertia_q, force_no_orth)
    [T, N] = size(x_rel);

    if nargin < 11
        force_no_orth = false;
    end

    if isempty(update_mix), update_mix = 1; end
    if isempty(max_turnover), max_turnover = Inf; end

    cum_wealth = ones(T, 1);
    daily_incre_fact = ones(T, 1);
    b_history = ones(N, T) / N;

    b_current = ones(N, 1) / N;
    b_prev = zeros(N, 1);
    run_ret = 1;

    for t = 1:T
        b_history(:, t) = b_current;

        turnover_t = sum(abs(b_current - b_prev));
        daily_incre = (x_rel(t, :) * b_current) * (1 - tran_cost / 2 * turnover_t);
        daily_incre_fact(t) = daily_incre;
        run_ret = run_ret * daily_incre;
        cum_wealth(t) = run_ret;

        b_prev = b_current .* x_rel(t, :)' / (x_rel(t, :) * b_current);

        if t < T

            if force_no_orth
                b_next_raw = IPT(p_close, x_rel, t, b_current, win_size, w_YAR, Q_factor, true);
            else
                b_next_raw = IPT(p_close, x_rel, t, b_current, win_size, w_YAR, Q_factor);
            end

            delta = b_next_raw - b_current;

            if isscalar(update_mix)
                alpha = update_mix;
            else
                alpha = update_mix(t);
            end

            if adaptive_inertia_q
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

end
