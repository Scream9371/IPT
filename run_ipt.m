function [cum_wealth, daily_incre_fact, b_history, L_history, yar_ubah_history] = run_ipt(data_path, weight_inspect_wins, risk_inspect_wins, tran_cost, reverse_factor, risk_factor, q_value, L_percentile)
    % Main execution function for IPT (Investment Potential Tracking) algorithm

    win_size = 5;

    load(data_path, 'data');
    [n_periods, ~] = size(data);

    yar_weights_full_wins = yar_weights(data, weight_inspect_wins);
    yar_weights_half_wins = yar_weights(data, floor(weight_inspect_wins / 2));

    ratio = ubah_price_ratio(data);

    yar_ubah_full_wins = yar_ubah(ratio(weight_inspect_wins - risk_inspect_wins + 1:n_periods), risk_inspect_wins);
    yar_ubah_half_wins = yar_ubah(ratio(floor(weight_inspect_wins / 2) - floor(risk_inspect_wins / 2) + 1:n_periods), floor(risk_inspect_wins / 2));

    % Adjust the dimensions to match by truncating the longer one
    yar_weights_long_size = size(yar_weights_full_wins, 1);
    yar_weights_short_size = size(yar_weights_half_wins, 1);
    yar_ubah_full_size = size(yar_ubah_full_wins, 1);
    yar_ubah_half_size = size(yar_ubah_half_wins, 1);

    % Use the minimum size to ensure dimensions match
    min_size = min([yar_weights_long_size, yar_weights_short_size, yar_ubah_full_size, yar_ubah_half_size]);

    if min_size > 0
        yar_weights_full_wins = yar_weights_full_wins(1:min_size, :);
        yar_weights_half_wins = yar_weights_half_wins(1:min_size, :);
        yar_ubah_full_wins = yar_ubah_full_wins(1:min_size, :);
        yar_ubah_half_wins = yar_ubah_half_wins(1:min_size, :);
    end

    % Double-check dimensions before calling active_function
    if size(yar_weights_full_wins, 1) ~= size(yar_weights_half_wins, 1) || ...
            size(yar_weights_full_wins, 1) ~= size(yar_ubah_full_wins, 1) || ...
            size(yar_weights_full_wins, 1) ~= size(yar_ubah_half_wins, 1)
        error('Dimension mismatch after adjustment: yar_weights_full_wins(%d), yar_weights_half_wins(%d), yar_ubah_full_wins(%d), yar_ubah_half_wins(%d)', ...
            size(yar_weights_full_wins, 1), size(yar_weights_half_wins, 1), ...
            size(yar_ubah_full_wins, 1), size(yar_ubah_half_wins, 1));
    end

    if nargin < 8 || isempty(L_percentile)
        L_percentile = 95;
    end

    yar_ubah_history = yar_ubah_full_wins(:, 1);
    L_history = compute_yar_percentile(yar_ubah_history, L_percentile);
    L_near_history = compute_yar_percentile(yar_ubah_half_wins(:, 1), L_percentile);

    try
        [w_YAR, Q_factor] = active_function(yar_weights_full_wins, yar_weights_half_wins, yar_ubah_full_wins, yar_ubah_half_wins, data, floor(weight_inspect_wins), reverse_factor, risk_factor, q_value, L_history, L_near_history);
    catch ME
        error('Error in active_function: %s', ME.message);
    end

    try
        [cum_wealth, daily_incre_fact, b_history] = IPT_run(data, win_size, tran_cost, w_YAR, Q_factor);
    catch ME
        error('Error in IPT_run: %s', ME.message);
    end

end
