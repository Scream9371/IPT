function [w] = compute_yar_weights(data, inspect_wins)
    [T, N] = size(data);
    return_rate_mean_total = ones(T - inspect_wins, N);

    for i = 1:T - inspect_wins
        return_rate_sample_mean = mean(data(i:inspect_wins + i - 1, :));
        return_rate_mean_total(i, :) = return_rate_sample_mean(1, :);
    end

    DR_total = ones(T - inspect_wins, N);

    for i = 1:T - inspect_wins
        negetive_date = zeros(1, N);
        x_minus_mean = data(i:inspect_wins + i - 1, :) - 1;
        %%x_minus_mean = data(i:inspect_wins+i-1,:)-mean(data(i:inspect_wins+i-1,:));
        for k = 1:inspect_wins
            for j = 1:N
                if x_minus_mean(k, j) > 0
                    x_minus_mean(k, j) = 0;
                else
                    negetive_date(1, j) = negetive_date(1, j) + 1;
                end
            end
        end

        downside_std = ones(1, N);
        yar_num = ones(1, N);
        x_minus_mean_sample = ones(1, inspect_wins);

        for j = 1:N
            for k = 1:inspect_wins
                x_minus_mean_sample(1, k) = (x_minus_mean(k, j)) ^ 2;
            end
            denom = max(negetive_date(1, j), 1);
            s = sum(x_minus_mean_sample);
            downside_std(1, j) = sqrt(s / denom);
            yar_num(1, j) = sqrt(s) / denom;
        end

        DR_total(i, :) = yar_num(1, :);
    end

    denom = return_rate_mean_total;
    denom(abs(denom) < 1e-12) = 1e-12;
    w = DR_total ./ denom;
end
