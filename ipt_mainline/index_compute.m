function [index] = index_compute(datasets)
    [datasets_T, datasets_N] = size(datasets);
    stock_price = ones(datasets_T, datasets_N);

    for i = 2:datasets_T
        stock_price(i, :) = stock_price(i - 1, :) .* datasets(i, :);
    end

    index = ones(datasets_T, 1);

    for i = 2:datasets_T
        index(i) = sum(stock_price(i, :)) / sum(stock_price(i - 1, :));
    end
