function [b_next] = IPT(p_close, x_rel, current_t, b_current, win_size, w_YAR, Q_factor)
    % IPT - Investment Potential Tracking algorithm for portfolio selection
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
    %   p_close       - n x m matrix of close price sequences for n time periods and m assets
    %   x_rel         - n x m matrix of price relative sequences (daily price ratios)
    %   current_t     - Current time step t (integer)
    %   b_current     - m x 1 vector of portfolio weights at time t
    %   win_size      - Window size for peak price calculation (integer)
    %   w_YAR         - n x m matrix of Yield-Adjusted Risk values
    %   Q_factor      - n x 1 vector of effect factor coefficients
    %
    % Output:
    %   b_next        - m x 1 vector of updated portfolio weights at time t+1

    epsilon = 100; % a parameter that controls the update step size
    % a = 0.5;

    nstk = size(x_rel, 2);

    if current_t < win_size + 1
        x_tplus1 = 1 .* (x_rel(current_t, :)) - Q_factor(current_t) .* w_YAR(current_t, :);
    else
        closebefore = p_close((current_t - win_size + 1):(current_t), :);
        closepredict = max(closebefore);

        x_tplus1 = 1 .* (closepredict ./ p_close(current_t, :)) - Q_factor(current_t) .* w_YAR(current_t, :);
    end

    onesd = ones(nstk, 1);
    x_tplus1_cent = (eye(nstk) - onesd * onesd' / nstk) * x_tplus1';

    if norm(x_tplus1_cent) ~= 0
        b_current = b_current + epsilon * x_tplus1_cent / norm(x_tplus1_cent);
    end

    b_next = simplex_projection_selfnorm2(b_current, 1);
