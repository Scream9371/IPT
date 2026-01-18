function run_full_diagnostics_all(varargin)
% run_full_diagnostics_all - Diagnose IPT implementation/logic across all datasets.
%
% What it checks per dataset:
%   1) PPT (fixed params) vs IPT with Q=0 (should be close if implementations align).
%   2) IPT full (three-state) using best IPT-specific params from fixed-summary.
%   3) Three-state activation statistics (state_code distribution and Q stats).
%
% Outputs:
%   Investment-potential-tracking/results_fixed_params/diagnostics_all.csv
%   Investment-potential-tracking/results_fixed_params/diagnostics_all.txt
%   Investment-potential-tracking/results_fixed_params/diagnostics_<dataset>.txt

    p = inputParser;
    addParameter(p, 'win_size', 5);
    addParameter(p, 'epsilon', 100);
    addParameter(p, 'tran_cost', 0.001);
    addParameter(p, 'summary_csv', fullfile(pwd, 'Investment-potential-tracking', 'results_fixed_params', 'ipt_fixed_log_wealth_summary.csv'));
    addParameter(p, 'L_smoothing_alpha', 0.2);
    parse(p, varargin{:});
    opts = p.Results;

    script_dir = fileparts(mfilename('fullpath'));
    base_dir = fileparts(script_dir);
    data_dir = fullfile(script_dir, 'Data Set');
    out_dir = fullfile(script_dir, 'results_fixed_params');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    if ~isfile(opts.summary_csv)
        error('Missing IPT fixed summary CSV: %s (run ipt_fixed_test first).', opts.summary_csv);
    end
    ipt_summary = readtable(opts.summary_csv);

    files = dir(fullfile(data_dir, '*.mat'));
    if isempty(files)
        error('No datasets found in %s', data_dir);
    end
    [~, order] = sort({files.name});
    files = files(order);

    ppt_dir = fullfile(base_dir, 'PPT');

    rows = struct('dataset', {}, 'T', {}, 'N', {}, ...
        'train_end', {}, 'val_start', {}, 'val_end', {}, 'test_start', {}, 'test_end', {}, ...
        'tran_cost', {}, 'win_size', {}, 'epsilon', {}, ...
        'ppt_test_wealth', {}, 'ipt_nostate_test_wealth', {}, 'ipt_full_test_wealth', {}, ...
        'absdiff_ppt_vs_ipt_nostate', {}, 'maxabs_wdiff_ppt_vs_ipt_nostate', {}, ...
        'meanabs_wdiff_ppt_vs_ipt_nostate', {}, ...
        'test_state1', {}, 'test_state2', {}, 'test_state3', {}, 'test_state4', {}, 'test_state5', {}, ...
        'test_mean_abs_Q', {});

    for i = 1:numel(files)
        dataset = erase(files(i).name, '.mat');
        data_path = fullfile(data_dir, files(i).name);
        S = load(data_path, 'data');
        data = S.data;
        [T, N] = size(data);

        train_end = floor(T * 0.6);
        val_end = floor(T * 0.8);
        val_start = train_end + 1;
        test_start = val_end + 1;
        test_end = T;

        fprintf('\n=== Diagnostics: %s (T=%d, N=%d) ===\n', dataset, T, N);

        % --- PPT fixed reference ---
        [ppt_daily_incre, ppt_weights] = simulate_ppt_full(ppt_dir, data, opts.win_size, opts.epsilon, opts.tran_cost);
        ppt_test_wealth = prod(ppt_daily_incre(test_start:test_end));

        % --- IPT "no-state" (Q=0) ---
        w0 = zeros(T, N);
        Q0 = zeros(T, 1);
        [~, ipt_daily_incre_nostate, b_hist_nostate] = IPT_run(data, opts.win_size, opts.tran_cost, w0, Q0, opts.epsilon);
        ipt_test_wealth_nostate = prod(ipt_daily_incre_nostate(test_start:test_end));
        ipt_weights_nostate = b_hist_nostate';

        maxabs_wdiff = max(abs(ppt_weights(:) - ipt_weights_nostate(:)));
        meanabs_wdiff = mean(abs(ppt_weights(:) - ipt_weights_nostate(:)));
        absdiff_wealth = abs(ppt_test_wealth - ipt_test_wealth_nostate);

        % --- IPT full (three-state) using fixed-summary best params ---
        row_idx = find(string(ipt_summary.dataset) == string(dataset), 1);
        if isempty(row_idx)
            fprintf('Skipping IPT full for %s: not found in %s\n', dataset, opts.summary_csv);
            ipt_test_wealth_full = nan;
            test_state = nan(1, 5);
            test_mean_abs_Q = nan;
        else
            weight_inspect_wins = ipt_summary.weight_inspect_wins(row_idx);
            risk_inspect_wins = ipt_summary.risk_inspect_wins(row_idx);
            L_percentile = ipt_summary.L_percentile(row_idx);
            q_value = ipt_summary.q_value(row_idx);
            reverse_factor = ipt_summary.reverse_factor(row_idx);
            risk_factor = ipt_summary.risk_factor(row_idx);

            p_close = ones(T, N);
            for t = 2:T
                p_close(t, :) = p_close(t - 1, :) .* data(t, :);
            end

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

            [~, ipt_daily_incre_full] = IPT_run(data, opts.win_size, opts.tran_cost, w_YAR, Q_factor, opts.epsilon);
            ipt_test_wealth_full = prod(ipt_daily_incre_full(test_start:test_end));

            test_codes = state_meta.state_code(test_start:test_end);
            test_state = nan(1, 5);
            for s = 1:5
                test_state(s) = mean(test_codes == s, 'omitnan');
            end
            test_mean_abs_Q = mean(abs(Q_factor(test_start:test_end)), 'omitnan');

            txt_path = fullfile(out_dir, sprintf('diagnostics_%s.txt', dataset));
            fid = fopen(txt_path, 'w');
            if fid ~= -1
                fprintf(fid, 'dataset=%s\n', dataset);
                fprintf(fid, 'T=%d, N=%d\n', T, N);
                fprintf(fid, 'fixed: tran_cost=%.6f, win_size=%d, epsilon=%.1f\n', opts.tran_cost, opts.win_size, opts.epsilon);
                fprintf(fid, 'split: train=1:%d, val=%d:%d, test=%d:%d\n', train_end, val_start, val_end, test_start, test_end);
                fprintf(fid, '\n[PPT fixed]\n');
                fprintf(fid, 'test_wealth=%.12f\n', ppt_test_wealth);
                fprintf(fid, '\n[IPT no-state (Q=0)]\n');
                fprintf(fid, 'test_wealth=%.12f\n', ipt_test_wealth_nostate);
                fprintf(fid, 'absdiff_vs_ppt=%.12f\n', absdiff_wealth);
                fprintf(fid, 'maxabs_weight_diff_vs_ppt=%.12g\n', maxabs_wdiff);
                fprintf(fid, 'meanabs_weight_diff_vs_ppt=%.12g\n', meanabs_wdiff);
                fprintf(fid, '\n[IPT full (three-state)]\n');
                fprintf(fid, 'params: weight_inspect_wins=%d, risk_inspect_wins=%d, L_percentile=%.1f, q=%.2f, reverse=%g, risk=%g\n', ...
                    weight_inspect_wins, risk_inspect_wins, L_percentile, q_value, reverse_factor, risk_factor);
                fprintf(fid, 'test_wealth=%.12f\n', ipt_test_wealth_full);
                fprintf(fid, 'test_mean_abs_Q=%.12g\n', test_mean_abs_Q);
                fprintf(fid, 'test_state_ratio: s1=%.6f, s2=%.6f, s3=%.6f, s4=%.6f, s5=%.6f\n', ...
                    test_state(1), test_state(2), test_state(3), test_state(4), test_state(5));
                fclose(fid);
            end
        end

        entry = struct();
        entry.dataset = dataset;
        entry.T = T;
        entry.N = N;
        entry.train_end = train_end;
        entry.val_start = val_start;
        entry.val_end = val_end;
        entry.test_start = test_start;
        entry.test_end = test_end;
        entry.tran_cost = opts.tran_cost;
        entry.win_size = opts.win_size;
        entry.epsilon = opts.epsilon;
        entry.ppt_test_wealth = ppt_test_wealth;
        entry.ipt_nostate_test_wealth = ipt_test_wealth_nostate;
        entry.ipt_full_test_wealth = ipt_test_wealth_full;
        entry.absdiff_ppt_vs_ipt_nostate = absdiff_wealth;
        entry.maxabs_wdiff_ppt_vs_ipt_nostate = maxabs_wdiff;
        entry.meanabs_wdiff_ppt_vs_ipt_nostate = meanabs_wdiff;
        entry.test_state1 = test_state(1);
        entry.test_state2 = test_state(2);
        entry.test_state3 = test_state(3);
        entry.test_state4 = test_state(4);
        entry.test_state5 = test_state(5);
        entry.test_mean_abs_Q = test_mean_abs_Q;
        rows(end + 1) = entry;

        fprintf('PPT test=%.6f, IPT(Q=0) test=%.6f, |diff|=%.6f, max|Δw|=%.3g\n', ...
            ppt_test_wealth, ipt_test_wealth_nostate, absdiff_wealth, maxabs_wdiff);
        if isfinite(ipt_test_wealth_full)
            fprintf('IPT full test=%.6f, mean|Q|_test=%.3g\n', ipt_test_wealth_full, test_mean_abs_Q);
        end
    end

    Tsum = struct2table(rows);
    csv_path = fullfile(out_dir, 'diagnostics_all.csv');
    writetable(Tsum, csv_path);

    txt_path = fullfile(out_dir, 'diagnostics_all.txt');
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

function [daily_incre, weights] = simulate_ppt_full(ppt_dir, data, win_size, epsilon, tran_cost)
    addpath(ppt_dir, '-begin');
    clear PPT PPT_run simplex_projection_selfnorm2

    [T, N] = size(data);
    close_price = ones(T, N);
    for i = 2:T
        close_price(i, :) = close_price(i - 1, :) .* data(i, :);
    end

    daily_port = ones(N, 1) / N;
    daily_port_o = zeros(N, 1);
    daily_incre = ones(T, 1);
    weights = zeros(T, N);

    for t = 1:T
        weights(t, :) = daily_port';
        daily_incre(t) = (data(t, :) * daily_port) * (1 - tran_cost / 2 * sum(abs(daily_port - daily_port_o)));
        daily_port_o = daily_port .* data(t, :)' / (data(t, :) * daily_port);
        if t < T
            [daily_port_n, ~, ~] = PPT(close_price, data, t, daily_port, win_size, epsilon);
            daily_port = daily_port_n;
        end
    end

    rmpath(ppt_dir);
    clear PPT PPT_run simplex_projection_selfnorm2
end
