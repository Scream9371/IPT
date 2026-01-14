function [YAR_ubah] = yar_ubah(ratio, inspect_wins)
    % yar_ubah - Calculate YAR (Yield-Adjusted Risk) factor for UBAH portfolio data.
    %
    %   [YAR_ubah] = yar_ubah(ratio, inspect_wins) computes the YAR factor
    %   (risk-adjusted factor) for portfolio adjustment using YAR methodology on
    %   UBAH portfolio price ratios as proposed in the IPT.
    %   Following equation:
    %   YAR_{t+1}^{ubah} = [√∑(min(P^{ubah}_t/P^{ubah}_{t-1} - 1, 0))² / n_negative] / [(1/n)∑r_t^i + 1]
    %
    %   Inputs:
    %     ratio          - Vector of UBAH portfolio price ratios (n × 1), where n is number of
    %                      time periods. Contains P^{ubah}_t/P^{ubah}_{t-1} ratios.
    %     inspect_wins   - Integer, window size for rolling calculation
    %
    %   Output:
    %     YAR_ubah       - Matrix of YAR factors (n - inspect_wins × 1) used for
    %                      portfolio risk management and adjustment

    [n_periods, m_assets] = size(ratio);
    mean_returns_total = ones(n_periods - inspect_wins, m_assets);

    for i = 1:n_periods - inspect_wins
        sample_mean_returns = mean(ratio(i:inspect_wins + i - 1, :)); % 1 × n vector
        mean_returns_total(i, :) = sample_mean_returns;
    end

    % Calculate ADV (Average Downside Volatility) - core component of YAR
    ADV_total = ones(n_periods - inspect_wins, m_assets);

    for i = 1:n_periods - inspect_wins
        negative_periods_count = zeros(1, m_assets);
        daily_returns = ratio(i:inspect_wins + i - 1, :) - 1;
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
                downside_volatility_sqrt(1, j) = sqrt(sum(downside_returns_sample) / negative_periods_count(1, j));
            else
                downside_volatility_sqrt(1, j) = 0; % Avoid division by zero
            end
        end

        ADV_total(i, :) = downside_volatility_sqrt(1, :);
    end

    YAR_ubah = ADV_total ./ mean_returns_total;
