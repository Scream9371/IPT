% Main execution script for IPT (Investment Potential Tracking) algorithm
% This script implements the complete IPT workflow:
% 1. Data loading and parameter initialization
% 2. YAR calculation for long-term and near-term windows
% 3. UBAH portfolio price ratio calculation
% 4. Three-state model selection strategy
% 5. IPT model execution with BP algorithm optimization

weight_inspect_wins = 252;
risk_inspect_wins = 84;
win_size = 5;
tran_cost = 0.001;

load('Data Set\djia.mat');
[n_periods, m_assets] = size(data);

yar_weights_full_wins = zeros(n_periods, m_assets);
yar_weights_value_full_wins = yar_weights(data, weight_inspect_wins);
yar_weights_full_wins(weight_inspect_wins + 1:n_periods, :) = yar_weights_value_full_wins(:, :);

yar_weights_half_wins = zeros(n_periods, m_assets);
yar_weights_value_half_wins = yar_weights(data, weight_inspect_wins / 2);
yar_weights_half_wins(weight_inspect_wins / 2 + 1:n_periods, :) = yar_weights_value_half_wins(:, :);

ratio = ubah_price_ratio(data);
reverse_factor = 5;
risk_factor = 5;

yar_ubah_full_wins = zeros(n_periods, 1);
yar_ubah_value_full_wins = yar_ubah(ratio(weight_inspect_wins - risk_inspect_wins + 1:n_periods, :), risk_inspect_wins);
yar_ubah_full_wins(weight_inspect_wins + 1:n_periods, 1) = yar_ubah_value_full_wins(:, 1);

yar_ubah_half_wins = zeros(n_periods, 1);
yar_ubah_value_half_wins = yar_ubah(ratio(weight_inspect_wins / 2 - risk_inspect_wins / 2 + 1:n_periods, :), risk_inspect_wins / 2);
yar_ubah_half_wins(weight_inspect_wins / 2 + 1:n_periods, 1) = yar_ubah_value_half_wins(:, 1);

[w_YAR, Q_factor] = active_function(yar_weights_value_full_wins, yar_weights_value_half_wins, yar_ubah_value_full_wins, yar_ubah_value_half_wins, data, weight_inspect_wins, reverse_factor, risk_factor);

[cum_wealth, daily_incre_fact, b_history] = IPT_run(data, win_size, tran_cost, w_YAR, Q_factor);
