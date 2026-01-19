function [YAR_weights] = yar_weights(data, inspect_wins)
    % yar_weights - Calculate per-asset YAR factors on a rolling window.
    %
    % Past-only alignment: YAR_weights(i,:) uses window [i+1 : i+inspect_wins],
    % which corresponds to time t = i + inspect_wins (paper-aligned).

    [n_periods, m_assets] = size(data);
    n_rows = n_periods - inspect_wins;
    if n_rows <= 0
        YAR_weights = zeros(0, m_assets);
        return;
    end

    YAR_weights = zeros(n_rows, m_assets);

    for i = 1:n_rows
        X = data((i + 1):(inspect_wins + i), :);
        u = X - 1;
        uN = min(u, 0);

        neg_cnt = sum(u < 0, 1);
        neg_cnt = max(neg_cnt, 1);

        ADV = sqrt(sum(uN .^ 2, 1) ./ neg_cnt);
        rbar = mean(X, 1);

        YAR_weights(i, :) = ADV ./ max(rbar, 1e-12);
    end
end
