% verify_highpot_latest.m
% Re-run the IPT_HighPot configuration using the FIXED codebase.
%
% Goal: Check if the "Rank 3" performance holds with correct code.
%
% Configuration:
%   Structure: No Qclip, No Inertia, Orthogonalized
%   Params: Smart Grid Selection (Q, Risk, Win) on Validation Set
%   Cost: 0.001

clear; clc;
base_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(base_dir, 'code'));

% Diagnostic checks
fprintf('Active Function Path: %s\n', which('active_function'));
fprintf('IPT Run Core Path:    %s\n', which('ipt_run_core'));
fprintf('IPT Update Path:      %s\n', which('IPT'));

dataset_names = {'djia', 'inv500', 'marpd', 'msci', 'nyse-n', 'nyse-o', 'sz50', 'tse'};
data_dir = fullfile(base_dir, '..', 'Data Set');

results_dir = fullfile(base_dir, 'results_verify_highpot');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

% Fixed Structural Params
p_struct = struct();
p_struct.epsilon = 100;
p_struct.tran_cost = 0.001;
p_struct.win_size = 5;
p_struct.update_mix = 1.0;      % No Inertia
p_struct.max_turnover = Inf;    % No Turnover Limit
p_struct.Q_clip_max = Inf;      % No Q Clipping
p_struct.adaptive_inertia_q = 0;
p_struct.force_no_orth = false; % Orthogonalized (Standard)
p_struct.L_smoothing_alpha = 0.2;
p_struct.Q_smoothing_alpha = 0;
p_struct.risk_inspect_wins = 21;
p_struct.L_percentile = 95;     % Standard

% Grid for Search
grid_win = [126, 252];
grid_q = [0.1, 0.2, 0.3]; 
grid_risk = [5, 10, 20];

final_cws = [];

fprintf('\n=== Verifying IPT_HighPot with CLEAN Code ===\n');

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
    
    % Validation window: Last 20% of Dev set
    val_len = floor(split_idx * 0.2); 
    val_start = split_idx - val_len + 1;
    val_end = split_idx;
    
    best_score = -Inf;
    best_p = p_struct;
    
    % --- Smart Grid Search ---
    for w = grid_win
        for q = grid_q
            for r = grid_risk
                % Generate Signals on Dev Set
                % Note: active_function needs full length inputs relative to 'data' passed
                % We pass dev_data
                dev_data = data(1:split_idx, :);
                ratio = ubah_price_ratio(dev_data);
                
                % Check window feasibility
                if w > size(dev_data, 1), continue; end
                
                yar_weights_long = yar_weights(dev_data, w);
                yar_weights_near = yar_weights(dev_data, floor(w/2));
                
                r = p_struct.risk_inspect_wins;
                r3 = max(2, floor(r / 3));
                start_long = w - r3 + 1;
                if start_long < 1, continue; end
                
                yar_ubah_long = yar_ubah(ratio(start_long:end, :), r3);
                half_r3 = max(2, floor(floor(r / 2) / 3));
                start_near = floor(w/2) - half_r3 + 1;
                yar_ubah_near = yar_ubah(ratio(start_near:end, :), half_r3);
                
                % L History
                L_raw = compute_yar_percentile(yar_ubah_long(:, 1), p_struct.L_percentile);
                L_history = ipt_smooth_series(L_raw, p_struct.L_smoothing_alpha);
                
                % Active Function
                [w_YAR, Q_factor] = active_function( ...
                    yar_weights_long, yar_weights_near, ...
                    yar_ubah_long, yar_ubah_near, ...
                    dev_data, w, ...
                    r, q, L_history);
                
                % Run Core
                [~, daily_ret_val] = ipt_run_core(dev_data, p_struct.win_size, p_struct.tran_cost, ...
                    w_YAR, Q_factor, p_struct.epsilon, p_struct.update_mix, p_struct.max_turnover, p_struct.adaptive_inertia_q, p_struct.force_no_orth);
                
                % Evaluate
                val_end_use = min(val_end, length(daily_ret_val));
                val_ret = daily_ret_val(val_start:val_end_use);
                score = prod(val_ret);
                
                if score > best_score
                    best_score = score;
                    best_p = p_struct;
                    best_p.win = w;
                    best_p.q = q;
                    best_p.risk = r;
                end
            end
        end
    end
    
    fprintf('Dataset %s Best: Win=%d, Q=%.1f, Risk=%d (Val Score: %.4f)\n', ...
        dname, best_p.win, best_p.q, best_p.risk, best_score);
        
    % --- Run on Test Set ---
    % Generate signals on FULL data
    ratio = ubah_price_ratio(data);
    yar_weights_long = yar_weights(data, best_p.win);
    yar_weights_near = yar_weights(data, floor(best_p.win/2));
    
    r = p_struct.risk_inspect_wins;
    r3 = max(2, floor(r / 3));
    start_long = best_p.win - r3 + 1;
    yar_ubah_long = yar_ubah(ratio(start_long:end, :), r3);
    
    % Fix index for ubah_near to match active_function expectation?
    % active_function logic: 
    % near_index = i + floor(win_long / 2);
    % It expects yar_ubah_near to be aligned such that yar_ubah_near(near_index) is valid.
    % Actually, my previous 'reproduce' script passed slices.
    % Let's look at `active_function` again.
    % It uses `yar_ubah_near` directly.
    % To be safe, let's just generate them with the standard helper functions 
    % which usually return aligned series or handle it.
    % `yar_ubah` returns T-win+1 rows.
    % `active_function` handles the indexing internally carefully.
    
    % Let's align exactly as in `run_ipt_fixed_test.m` or similar working scripts.
    % Re-reading `run_ipt_fixed_test.m`:
    % yar_ubah_near = yar_ubah(ratio(start_near:T, :), half_risk);
    
    half_r3 = max(2, floor(floor(r / 2) / 3));
    start_near = floor(best_p.win/2) - half_r3 + 1;
    yar_ubah_near = yar_ubah(ratio(start_near:end, :), half_r3);
    
    L_raw = compute_yar_percentile(yar_ubah_long(:, 1), p_struct.L_percentile);
    L_history = ipt_smooth_series(L_raw, p_struct.L_smoothing_alpha);
    
    [w_YAR, Q_factor] = active_function( ...
        yar_weights_long, yar_weights_near, ...
        yar_ubah_long, yar_ubah_near, ...
        data, best_p.win, ...
        best_p.risk, best_p.q, L_history);
        
    [cum_wealth, daily_ret] = ipt_run_core(data, p_struct.win_size, p_struct.tran_cost, ...
        w_YAR, Q_factor, p_struct.epsilon, p_struct.update_mix, p_struct.max_turnover, p_struct.adaptive_inertia_q, p_struct.force_no_orth);
        
    test_ret = daily_ret(test_idx:end);
    final_wealth = prod(test_ret);
    
    fprintf('  Test Wealth: %.4f\n', final_wealth);
    final_cws(end+1) = final_wealth;
    
    save(fullfile(results_dir, ['ipt_verify-' dname '.mat']), 'test_ret', 'final_wealth', 'best_p');
end

fprintf('\nSummary of Verify Run:\n');
disp(final_cws);
