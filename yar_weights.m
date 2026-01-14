function [YAR_weights] = yar_weights(data, inspect_wins)
    % yar_weights - Calculate per-asset YAR factors on a rolling window.
    %
    % This version computes downside risk per asset independently to avoid
    % cross-asset contamination in ADV. The window uses only historical data
    % [i : i+inspect_wins-1], which corresponds to information up to t-1 when
    % used by active_function at time t.

    [n_periods, m_assets] = size(data);
    n_rows = n_periods - inspect_wins;
    if n_rows <= 0
        YAR_weights = zeros(0, m_assets);
        return;
    end

    YAR_weights = zeros(n_rows, m_assets);

    for i = 1:n_rows
        X = data(i:(inspect_wins + i - 1), :);
        u = X - 1;
        uN = min(u, 0);

        neg_cnt = sum(u < 0, 1);
        neg_cnt = max(neg_cnt, 1);

        ADV = sqrt(sum(uN .^ 2, 1) ./ neg_cnt);
        rbar = mean(X, 1);

        YAR_weights(i, :) = ADV ./ max(rbar, 1e-12);
    end
end
