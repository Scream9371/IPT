function [ratio] = ubah_price_ratio(data)
    % ubah_price_ratio calculates UBAH portfolio price ratios for YAR calculation.
    %
    % This function computes P^{ubah}_t/P^{ubah}_{t-1} ratios for UBAH strategy,
    % which are used as input data for calculating Yield-Adjusted Risk (YAR) under
    % the UBAH model.
    %
    % Input:
    %   data   - 2D double array of size n × m, where n is number of time periods
    %          and m is number of assets. Contains the relative price of each asset.
    %
    % Output:
    %   ratio  - 1D array of size n × 1 containing UBAH portfolio price ratios
    %           P^{ubah}_t/P^{ubah}_{t-1} for each time period under the UBAH strategy.

    [n_periods, m_assets] = size(data);
    stock_price = ones(n_periods, m_assets);

    for i = 2:n_periods
        stock_price(i, :) = stock_price(i - 1, :) .* data(i, :);
    end

    ratio = ones(n_periods, 1);

    for i = 2:n_periods
        ratio(i) = sum(stock_price(i, :)) / sum(stock_price(i - 1, :));
    end
