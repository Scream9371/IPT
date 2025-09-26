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

w_full_year = zeros(n_periods, m_assets);
w_value_full_year = yar_weights(data, weight_inspect_wins);
w_full_year(weight_inspect_wins + 1:n_periods, :) = w_value_full_year(:, :);

w_half_year = zeros(n_periods, m_assets);
w_value_half_year = yar_weights(data, weight_inspect_wins / 2);
w_half_year(weight_inspect_wins / 2 + 1:n_periods, :) = w_value_half_year(:, :);

ratio = ubah_price_ratio(data);
reverse_factor = 5;
risk_factor = 5;

active_factor_full_year = zeros(n_periods, 1);
active_factor_value_full_year = yar_ubah(ratio(weight_inspect_wins - risk_inspect_wins + 1:n_periods, :), risk_inspect_wins);
active_factor_full_year(weight_inspect_wins + 1:n_periods, 1) = active_factor_value_full_year(:, 1);

active_factor_half_year = zeros(n_periods, 1);
active_factor_value_half_year = yar_ubah(ratio(weight_inspect_wins / 2 - risk_inspect_wins / 2 + 1:n_periods, :), risk_inspect_wins / 2);
active_factor_half_year(weight_inspect_wins / 2 + 1:n_periods, 1) = active_factor_value_half_year(:, 1);

[w_YAR, Q_factor] = active_function(w_value_full_year, w_value_half_year, active_factor_value_full_year, active_factor_value_half_year, data, weight_inspect_wins, reverse_factor, risk_factor);

[cum_wealth, daily_incre_fact, b_history] = IPT_run(data, win_size, tran_cost, w_YAR, Q_factor);
