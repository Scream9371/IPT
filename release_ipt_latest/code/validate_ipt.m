function validate_ipt()
    % VALIDATE_IPT  Perform strict equivalence check: IPT(Q=0) vs Standard PPT.
    %
    % This script verifies that when IPT is configured with:
    %   - Q_factor = 0 (everywhere)
    %   - update_mix = 1.0 (no inertia)
    %   - max_turnover = Inf (no constraint)
    %   - Same epsilon and win_size
    %
    % It produces BITWISE IDENTICAL portfolio weights to the standard 'PPT.m' implementation.

    clc;
    script_dir = fileparts(mfilename('fullpath'));

    % Path setup
    ppt_dir = fullfile(script_dir, '..', '..', '..', 'PPT');

    if ~exist(ppt_dir, 'dir')
        ppt_dir = fullfile(script_dir, '..', '..', '..', 'PPT&RPPT_two_wins'); % Fallback
    end

    if ~exist(ppt_dir, 'dir')
        error('Cannot find PPT implementation directory.');
    end

    addpath(ppt_dir);
    addpath(script_dir); % Ensure local ipt_run_core is found

    fprintf('Using PPT from: %s\n', which('PPT'));
    fprintf('Using ipt_run_core from: %s\n', which('ipt_run_core'));

    % Load Data
    data_dir = fullfile(script_dir, '..', '..', 'Data Set');
    dataset_name = 'msci.mat';
    data_path = fullfile(data_dir, dataset_name);

    if ~isfile(data_path)
        files = dir(fullfile(data_dir, '*.mat'));
        data_path = fullfile(data_dir, files(1).name);
    end

    fprintf('Validating on dataset: %s\n', data_path);
    S = load(data_path, 'data');
    data = S.data;
    [T, N] = size(data);

    % Common Params
    win_size = 5;
    epsilon = 100;
    tran_cost = 0.001;

    % --- Run 1: Standard PPT ---
    p_close = ones(T, N);

    for t = 2:T
        p_close(t, :) = p_close(t - 1, :) .* data(t, :);
    end

    ppt_b = zeros(N, T);
    ppt_b(:, 1) = ones(N, 1) / N;
    b_curr = ones(N, 1) / N;

    % PPT Loop (reproducing PPT_run logic but using the function)
    for t = 1:T - 1
        % Note: PPT function takes 'tplus1' as the target time to predict FOR.
        % Usually we call it at time 't' to predict 't+1'.
        % The PPT function signature is PPT(..., tplus1, ...).
        % Inside PPT: if tplus1 < win_size+1, it uses data(tplus1, :).
        % This implies 'tplus1' is the index of the day we just observed?
        % No, let's look at PPT code:
        % x_tplus1 = closepredict ./ close_price(tplus1, :)
        % This calculates the return vector for the update step.
        % So 'tplus1' is actually 'current_t' in IPT notation.

        % In IPT: current_t is the time of the update.
        % In PPT: the 3rd arg is 'tplus1'.
        % Let's align them.

        % Update portfolio for next day
        [b_next, ~, ~] = PPT(p_close, data, t, b_curr, win_size, epsilon);
        b_curr = b_next;
        ppt_b(:, t + 1) = b_curr;
    end

    % --- Run 2: IPT (Q=0) ---
    w_YAR_dummy = zeros(T, N);
    Q_factor_dummy = zeros(T, 1);

    % Note: ipt_run_core calls IPT(..., t, ...)
    % In IPT.m: r_hat = ... / p_close(current_t, :)
    % So 'current_t' in IPT corresponds to 'tplus1' in PPT.
    % They are aligned.

    [~, ~, ipt_b] = ipt_run_core(data, win_size, tran_cost, w_YAR_dummy, Q_factor_dummy, epsilon, 1.0, Inf);

    % ipt_b is N x T

    % --- Compare ---
    diff = abs(ppt_b - ipt_b);
    max_diff = max(diff(:));

    fprintf('\nMax difference in portfolio weights: %.6e\n', max_diff);

    if max_diff < 1e-10
        fprintf('✅ PASS: IPT(Q=0) is strictly equivalent to PPT.\n');
    else
        fprintf('❌ FAIL: Discrepancy detected.\n');
        [r, c] = find(diff > 1e-10, 1);
        fprintf('First divergence at t=%d, asset=%d (PPT=%.4f, IPT=%.4f)\n', c, r, ppt_b(r, c), ipt_b(r, c));
    end

    rmpath(ppt_dir);
end
