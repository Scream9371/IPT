% run_highpot_safe_ablation.m
% Search for a robust "HighPot_Safe" configuration using the FIXED codebase.
%
% Hypothesis:
%   - Conditional Orthogonalization (already fixed) protects strong trends (NYSE-O).
%   - Inertia (Mix < 1) and Q_clip (e.g. 10) are needed to rescue volatile markets (MSCI, TSE).
%
% Search Space:
%   - Q: [0.1, 0.2, 0.3]
%   - Risk: [5, 10, 20]
%   - Win: [126, 252]
%   - Mix: [0.5, 1.0] (0.5=Inertia, 1.0=No Inertia)
%   - Clip: [10, Inf] (10=Clipped, Inf=No Clip)
%
% Codebase: release_ipt_latest (FIXED)

clear; clc;
base_dir = fileparts(mfilename('fullpath'));
addpath(base_dir); % Add current dir (code) to path

dataset_names = {'djia', 'inv500', 'marpd', 'msci', 'nyse-n', 'nyse-o', 'sz50', 'tse'};
data_dir = fullfile(base_dir, '..', '..', 'Data Set');

results_dir = fullfile(base_dir, 'results_highpot_safe');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

% Fixed Structural Params
p_struct = struct();
p_struct.epsilon = 100;
p_struct.tran_cost = 0.001;
p_struct.win_size = 5;
p_struct.max_turnover = Inf;
p_struct.adaptive_inertia_q = 0; % Keep off for now to isolate standard inertia
p_struct.force_no_orth = false; % CONDITIONAL ORTHOGONALIZATION (Standard)
p_struct.L_smoothing_alpha = 0.2;
p_struct.Q_smoothing_alpha = 0;
p_struct.risk_inspect_wins = 21;
p_struct.L_percentile = 95;

% Expanded Grid
grid_win = [126, 252];
grid_q = [0.1, 0.2, 0.3];
grid_risk = [5, 10, 20];
grid_mix = [0.5, 1.0];
grid_clip = [10, Inf];

final_cws = [];

fprintf('\n=== HighPot_Safe Ablation (Mix/Clip Search) ===\n');

for di = 1:numel(dataset_names)
    dname = dataset_names{di};
    data_path = fullfile(data_dir, [dname '.mat']);
    if ~isfile(data_path), continue; end

    S = load(data_path, 'data');
    data = S.data;
    [T, N] = size(data);

    % Split: Dev 60% / Test 40%
    split_idx = floor(T * 0.6);
    test_idx = split_idx + 1;

    % Validation: Last 20% of Dev
    val_len = floor(split_idx * 0.2);
    val_start = split_idx - val_len + 1;
    val_end = split_idx;

    best_score = -Inf;
    best_p = p_struct;

    % --- Grid Search ---
    % Pre-calculate signal cache to speed up
    % Since Q/Risk/Win affect signals, we loop them outer.
    % Mix/Clip only affect execution, loop them inner.

    for w = grid_win
        % Dev data for signal generation
        dev_data = data(1:split_idx, :);
        if w > size(dev_data, 1), continue; end

        ratio = ubah_price_ratio(dev_data);
        yar_weights_long = yar_weights(dev_data, w);
        yar_weights_near = yar_weights(dev_data, floor(w / 2));

        r = p_struct.risk_inspect_wins;
        r3 = max(2, floor(r / 3)); % ALIGNMENT FIX
        start_long = w - r3 + 1;
        if start_long < 1, continue; end

        yar_ubah_long = yar_ubah(ratio(start_long:end, :), r3);

        half_r3 = max(2, floor(floor(r / 2) / 3)); % ALIGNMENT FIX
        half_weight = floor(w / 2);
        start_near = half_weight - half_r3 + 1;
        yar_ubah_near = yar_ubah(ratio(start_near:end, :), half_r3);

        L_raw = compute_yar_percentile(yar_ubah_long(:, 1), p_struct.L_percentile);
        L_history = ipt_smooth_series(L_raw, p_struct.L_smoothing_alpha);

        for q = grid_q

            for risk = grid_risk
                % Generate Signals
                [w_YAR, Q_factor_raw] = active_function( ...
                    yar_weights_long, yar_weights_near, ...
                    yar_ubah_long, yar_ubah_near, ...
                    dev_data, w, ...
                    risk, q, L_history);

                for clip = grid_clip
                    % Apply Clip
                    if isinf(clip)
                        Q_factor = Q_factor_raw;
                    else
                        Q_factor = max(min(Q_factor_raw, clip), -clip);
                    end

                    for mix = grid_mix
                        % Run Core
                        [~, daily_ret_val] = ipt_run_core(dev_data, p_struct.win_size, p_struct.tran_cost, ...
                            w_YAR, Q_factor, p_struct.epsilon, mix, p_struct.max_turnover, p_struct.adaptive_inertia_q, p_struct.force_no_orth);

                        % Evaluate
                        val_end_use = min(val_end, length(daily_ret_val));
                        val_ret = daily_ret_val(val_start:val_end_use);
                        score = prod(val_ret);

                        if score > best_score
                            best_score = score;
                            best_p = p_struct;
                            best_p.win = w;
                            best_p.q = q;
                            best_p.risk = risk;
                            best_p.mix = mix;
                            best_p.clip = clip;
                        end

                    end

                end

            end

        end

    end

    fprintf('Dataset %s Best: Win=%d, Q=%.1f, Risk=%d, Mix=%.1f, Clip=%.0f (Val: %.4f)\n', ...
        dname, best_p.win, best_p.q, best_p.risk, best_p.mix, best_p.clip, best_score);

    % --- Run on Test Set ---
    ratio = ubah_price_ratio(data);
    yar_weights_long = yar_weights(data, best_p.win);
    yar_weights_near = yar_weights(data, floor(best_p.win / 2));

    r = p_struct.risk_inspect_wins;
    r3 = max(2, floor(r / 3));
    start_long = best_p.win - r3 + 1;
    yar_ubah_long = yar_ubah(ratio(start_long:end, :), r3);

    half_r3 = max(2, floor(floor(r / 2) / 3));
    half_weight = floor(best_p.win / 2);
    start_near = half_weight - half_r3 + 1;
    yar_ubah_near = yar_ubah(ratio(start_near:end, :), half_r3);

    L_raw = compute_yar_percentile(yar_ubah_long(:, 1), p_struct.L_percentile);
    L_history = ipt_smooth_series(L_raw, p_struct.L_smoothing_alpha);

    [w_YAR, Q_factor_raw] = active_function( ...
        yar_weights_long, yar_weights_near, ...
        yar_ubah_long, yar_ubah_near, ...
        data, best_p.win, ...
        best_p.risk, best_p.q, L_history);

    if isinf(best_p.clip)
        Q_factor = Q_factor_raw;
    else
        Q_factor = max(min(Q_factor_raw, best_p.clip), -best_p.clip);
    end

    [cum_wealth, daily_ret] = ipt_run_core(data, p_struct.win_size, p_struct.tran_cost, ...
        w_YAR, Q_factor, p_struct.epsilon, best_p.mix, p_struct.max_turnover, p_struct.adaptive_inertia_q, p_struct.force_no_orth);

    test_ret = daily_ret(test_idx:end);
    final_wealth = prod(test_ret);

    fprintf('  Test Wealth: %.4f\n', final_wealth);
    final_cws(end + 1) = final_wealth;

    save(fullfile(results_dir, ['ipt_safe-' dname '.mat']), 'test_ret', 'final_wealth', 'best_p');
end

fprintf('\nSummary of HighPot_Safe Run:\n');
disp(final_cws);
