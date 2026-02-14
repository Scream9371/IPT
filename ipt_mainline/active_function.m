function [w, r] = active_function(w_full_year, w_half_year, active_factor_full_year, active_factor_half_year, datasets, inspect_wins, reverse_factor, risk_factor, L, q)
    [datasets_T, datasets_N] = size(datasets);
    w = zeros(datasets_T, datasets_N);
    r = zeros(datasets_T, 1);

    th_L1 = q * L / 2;
    th_L2 = q * L;
    th_N1 = (1 - q) * L;
    th_N2 = (1 - q / 2) * L;

    for i = 1:datasets_T - inspect_wins

        if active_factor_full_year(i) <= th_L1
            r(i + inspect_wins) = -2 * reverse_factor;
            w(i + inspect_wins, :) = w_full_year(i, :);
        elseif active_factor_full_year(i) <= th_L2
            r(i + inspect_wins) = -reverse_factor;
            w(i + inspect_wins, :) = w_full_year(i, :);
        else

            if active_factor_half_year(i + inspect_wins / 2) <= th_N1
                r(i + inspect_wins) = 0;
                w(i + inspect_wins, :) = w_half_year(i + inspect_wins / 2, :);
            elseif active_factor_half_year(i + inspect_wins / 2) <= th_N2
                r(i + inspect_wins) = risk_factor;
                w(i + inspect_wins, :) = w_half_year(i + inspect_wins / 2, :);
            else
                r(i + inspect_wins) = 2 * risk_factor;
                w(i + inspect_wins, :) = w_half_year(i + inspect_wins / 2, :);
            end

        end

    end
