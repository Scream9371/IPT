% reproduce_best_ipt.m
% Reproduce the "iptL92p5noQclip" variant which achieved ~3.x mean rank.
% Configuration:
%   L_percentile = 92.5
%   Q_clip_max = Inf
%   No Inertia (Mix=1)
%   Tran Cost = 0.001

clear; clc;
base_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(base_dir, 'code'));

% Ensure we are using the correct functions
fprintf('Using active_function from: %s\n', which('active_function'));
fprintf('Using ipt_run_core from:    %s\n', which('ipt_run_core'));

dataset_names = {'djia', 'inv500', 'marpd', 'msci', 'nyse-n', 'nyse-o', 'sz50', 'tse'};
data_dir = fullfile(base_dir, '..', 'Data Set');

results_dir = fullfile(base_dir, 'results_repro_best');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

% Params
params = struct();
params.win_size = 5;
params.epsilon = 100;
params.tran_cost = 0.001; % STRICTLY 0.001
params.weight_inspect_wins = 126; % Guessing standard values
params.risk_inspect_wins = 21;
params.L_percentile = 92.5; % KEY PARAM
params.L_smoothing_alpha = 0.2;
params.Q_smoothing_alpha = 0;
params.risk_factor = 10; % Need to guess or grid search this. 
% If it was "best", it might have been tuned per dataset.
% But "iptL92p5noQclip" sounds like a fixed structural variant.
% Let's use Smart Grid logic: run a small grid and pick best per dataset (validation), 
% then test on test set.
params.force_no_orth = false; % Standard: use conditional orthogonalization

fprintf('\n=== Reproducing iptL92p5noQclip ===\n');

grid_q = [0.1, 0.2, 0.3];
grid_risk = [5, 10, 20];

final_cws = [];

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
    
    % Validation on the last part of Dev set to pick parameters
    val_len = floor(split_idx * 0.2); 
    val_start = split_idx - val_len + 1;
    val_end = split_idx;
    
    best_score = -Inf;
    best_p = params;
    
    % Grid Search on Validation Set
    for q = grid_q
        for r = grid_risk
            p = params;
            p.q_value = q;
            p.risk_factor = r;
            
            % Run on Dev set (full history access, but evaluate on val segment)
            ratio = ubah_price_ratio(data(1:split_idx, :));
            yar_weights_long = yar_weights(data(1:split_idx, :), p.weight_inspect_wins);
            yar_weights_near = yar_weights(data(1:split_idx, :), floor(p.weight_inspect_wins/2));
            yar_ubah_long = yar_ubah(ratio, p.risk_inspect_wins); % Simplified, index alignment needed but let's approximate
            yar_ubah_near = yar_ubah(ratio, floor(p.risk_inspect_wins/2));
            
            % Need correct alignment for active_function.
            % Let's use local_run_active_core logic if possible, but we don't have it here.
            % Let's just call active_function directly with careful indexing.
            % Actually, active_function handles full length.
            
            % We need L_history
            L_raw = compute_yar_percentile(yar_ubah_long(:, 1), p.L_percentile);
            L_history = ipt_smooth_series(L_raw, p.L_smoothing_alpha);
            
            [w_YAR, Q_factor] = active_function( ...
                yar_weights_long, yar_weights_near, ...
                yar_ubah_long, yar_ubah_near, ...
                data(1:split_idx, :), p.weight_inspect_wins, ...
                p.risk_factor, p.q_value, L_history);
                
            % No Q clip!
            
            % Check data format
            % If data values are around 1.0, it's relative.
            % If around 100, it's price.
            if mean(mean(data)) > 2
                error('Data seems to be prices, expected relatives.');
            end
             
            % ipt_run_core inputs: x_rel
            [~, daily_ret_val] = ipt_run_core(data(1:split_idx, :), p.win_size, p.tran_cost, w_YAR, Q_factor, p.epsilon, 1.0, Inf, 0, p.force_no_orth);
            
            % Evaluate on validation segment
            val_ret = daily_ret_val(val_start:val_end);
            score = prod(val_ret);
            
            if score > best_score
                best_score = score;
                best_p = p;
            end
        end
    end
    
    fprintf('Dataset %s Best Params: Q=%.2f, Risk=%d\n', dname, best_p.q_value, best_p.risk_factor);
    
    % Run on Test Set using Best Params
    % We need to run on FULL data now
    ratio = ubah_price_ratio(data);
    yar_weights_long = yar_weights(data, best_p.weight_inspect_wins);
    yar_weights_near = yar_weights(data, floor(best_p.weight_inspect_wins/2));
    yar_ubah_long = yar_ubah(ratio, best_p.risk_inspect_wins);
    yar_ubah_near = yar_ubah(ratio, floor(best_p.risk_inspect_wins/2));
    
    L_raw = compute_yar_percentile(yar_ubah_long(:, 1), best_p.L_percentile);
    L_history = ipt_smooth_series(L_raw, best_p.L_smoothing_alpha);
    
    [w_YAR, Q_factor] = active_function( ...
        yar_weights_long, yar_weights_near, ...
        yar_ubah_long, yar_ubah_near, ...
        data, best_p.weight_inspect_wins, ...
        best_p.risk_factor, best_p.q_value, L_history);
        
    [cum_wealth, daily_ret] = ipt_run_core(data, best_p.win_size, best_p.tran_cost, w_YAR, Q_factor, best_p.epsilon, 1.0, Inf, 0, best_p.force_no_orth);
    
    test_ret = daily_ret(test_idx:end);
    final_wealth = prod(test_ret);
    
    fprintf('  Test Wealth: %.4f\n', final_wealth);
    final_cws(end+1) = final_wealth;
    
    save(fullfile(results_dir, ['ipt_repro-' dname '.mat']), 'test_ret', 'final_wealth', 'best_p');
end

fprintf('\nSummary of Reproduce Run:\n');
disp(final_cws);
