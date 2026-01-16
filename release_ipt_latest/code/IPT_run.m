function [cum_wealth, daily_incre_fact, b_history] = IPT_run(x_rel, win_size, trans_cost, w_YAR, Q_factor)
    % IPT_run - Main execution of Investment Potential Tracking strategy
    %
    % This function implements the core Investment Potential Tracking (IPT) algorithm,
    % an improved version of Peak Price Tracking (PPT)[1]. It dynamically adjusts
    % portfolio weights based on asset performance trends and risk factors.
    %
    % References:
    % [1] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. "A peak price tracking
    %     based learning system for portfolio selection", IEEE Transactions on Neural Networks and Learning Systems, 2017. Accepted.
    % [2] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. "Radial basis functions
    %     with adaptive input and composite trend representation for portfolio selection",
    %     IEEE Transactions on Neural Networks and Learning Systems, 2018. Accepted.
    % [3] Pei-Yi Yang, Zhao-Rong Lai*, Xiaotian Wu, Liangda Fang. "Trend Representation
    %     Based Log-density Regularization System for Portfolio Optimization",
    %     Pattern Recognition, vol. 76, pp. 14-24, Apr. 2018.
    % [4] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra. "Efficient
    %     projections onto the l1-ball for learning in high dimensions", in
    %     Proceedings of the International Conference on Machine Learning (ICML 2008), 2008.
    % [5] B. Li, D. Sahoo, and S. C. H. Hoi. Olps: a toolbox for on-line portfolio selection.
    %     Journal of Machine Learning Research, 17, 2016.
    %
    % Inputs:
    %   x_rel            - n x m matrix of price relatives (daily returns)
    %   win_size         - Lookback window size for peak price
    %   trans_cost       - Transaction cost rate (e.g., 0.001 = 0.1%)
    %   w_YAR            - n x m matrix of Yield-Adjusted Risk (YAR) values
    %   Q_factor         - n x 1 vector of effect factor coefficients
    %
    % Outputs:
    %   cum_wealth       - n x 1 cumulative wealth curve
    %   daily_incre_fact - n x 1 daily increasing factors
    %   b_history        - m x n matrix of portfolio weights over time

    [n_periods, m_assets] = size(x_rel);

    cum_wealth = ones(n_periods, 1);
    daily_incre_fact = ones(n_periods, 1);

    b_current = ones(m_assets, 1) / m_assets;
    b_history = ones(m_assets, n_periods) / m_assets;
    b_prev = zeros(m_assets, 1);

    p_close = ones(n_periods, m_assets);

    for i = 2:n_periods
        p_close(i, :) = p_close(i - 1, :) .* x_rel(i, :);
    end

    run_ret = 1;

    for t = 1:n_periods

        b_history(:, t) = b_current;
        daily_incre_fact(t, 1) = (x_rel(t, :) * b_current) * (1 - trans_cost / 2 * sum(abs(b_current - b_prev)));

        run_ret = run_ret * daily_incre_fact(t, 1);
        cum_wealth(t) = run_ret;

        b_prev = b_current .* x_rel(t, :)' / (x_rel(t, :) * b_current);

        if (t < n_periods)
            [b_next] = IPT(p_close, x_rel, t, b_current, win_size, w_YAR, Q_factor);

            b_current = b_next;

        end

    end
