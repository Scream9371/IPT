function [YAR_ubah] = yar_ubah(ratio, inspect_wins)
    % yar_ubah - Calculate UBAH YAR factor on a rolling window.
    %
    % Past-only alignment: YAR_ubah(i) uses window [i+1 : i+inspect_wins],
    % which corresponds to time t = i + inspect_wins (paper-aligned).

    [n_periods, n_cols] = size(ratio);
    n_rows = n_periods - inspect_wins;
    if n_rows <= 0
        YAR_ubah = zeros(0, n_cols);
        return;
    end

    YAR_ubah = zeros(n_rows, n_cols);

    for i = 1:n_rows
        X = ratio((i + 1):(inspect_wins + i), :);
        u = X - 1;
        uN = min(u, 0);

        neg_cnt = sum(u < 0, 1);
        neg_cnt = max(neg_cnt, 1);

        ADV = sqrt(sum(uN .^ 2, 1) ./ neg_cnt);
        rbar = mean(X, 1);

        YAR_ubah(i, :) = ADV ./ max(rbar, 1e-12);
    end
end
