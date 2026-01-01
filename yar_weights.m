function [YAR_weights] = yar_weights(data, inspect_wins)
    % yar_weights - Calculate portfolio weights based on Yield-Adjusted Risk (YAR) for stock price data.
    %
    %   [YAR_weights] = yar_weights(data, inspect_wins) computes portfolio weights using
    %   Yield-Adjusted Risk methodology on relative stock price data. The function
    %   calculates YAR using downside risk adjusted by mean returns as proposed in IPT.
    %   Following equation:
    %   YAR_{t+1}^i = ADV^i_{t+1} / [(1/n)∑r_t^i + 1], where ADV^i_{t+1} = √[∑(min(r_t^i - 1, 0))²] / n_negative.
    %
    %   Inputs:
    %     data          - Matrix of relative stock prices (n × m), where n is number of
    %                     time periods and m is number of assets
    %     inspect_wins  - Integer, window size for rolling calculation
    %
    %   Output:
    %     YAR_weights   - Matrix of Yield-Adjusted Risk weights (n - inspect_wins × m) representing
    %                     the risk-adjusted investment allocation for each asset

    [n_periods, m_assets] = size(data);
    mean_returns_total = ones(n_periods - inspect_wins, m_assets);

    for i = 1:n_periods - inspect_wins
        sample_mean_returns = mean(data(i:inspect_wins + i - 1, :));
        mean_returns_total(i, :) = sample_mean_returns;
    end
    
    % Calculate ADV (Average Downside Volatility) - core component of YAR
    ADV_total = ones(n_periods - inspect_wins, m_assets);

    for i = 1:n_periods - inspect_wins
        negative_periods_count = zeros(1, m_assets);
        daily_returns = data(i:inspect_wins + i - 1, :) - 1;
        for k = 1:inspect_wins

            for j = 1:m_assets

                if daily_returns(k, j) > 0
                    daily_returns(k, j) = 0;
                else
                    negative_periods_count(1, j) = negative_periods_count(1, j) + 1;
                end

            end

        end

        downside_volatility_sqrt = ones(1, m_assets);
        downside_returns_sample = ones(1, inspect_wins);

        for j = 1:m_assets

            for k = 1:inspect_wins
                downside_returns_sample(1, k) = (daily_returns(k, j)) ^ 2;
            end

            % ADV calculation
            if negative_periods_count(1, j) > 0
                downside_volatility_sqrt(1, j) = sqrt(sum(downside_returns_sample)) / (negative_periods_count(1, j));
            else
                downside_volatility_sqrt(1, j) = 0; % Avoid division by zero
            end
        end

        ADV_total(i, :) = downside_volatility_sqrt(1, :);
    end

    YAR_weights = ADV_total ./ mean_returns_total;
