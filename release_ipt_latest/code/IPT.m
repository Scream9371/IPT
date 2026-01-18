function [b_next] = IPT(p_close, x_rel, current_t, b_current, win_size, w_YAR, Q_factor, epsilon)
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

    if nargin < 8 || isempty(epsilon)
        % Default to 100 if not provided, but allow external control
        epsilon = 100;
    end
    % a = 0.5;

    nstk = size(x_rel, 2);

    if current_t < win_size + 1
        r_hat = x_rel(current_t, :);
    else
        closebefore = p_close((current_t - win_size + 1):(current_t), :);
        closepredict = max(closebefore);
        r_hat = closepredict ./ p_close(current_t, :);
    end

    e_hat = Q_factor(current_t) .* w_YAR(current_t, :);
    r_c = r_hat - mean(r_hat);
    e_c = e_hat - mean(e_hat);
    scale = min(1, norm(r_c, 2) / (norm(e_c, 2) + 1e-12));
    x_tplus1 = r_hat - scale * e_hat;

    onesd = ones(nstk, 1);
    x_tplus1_cent = (eye(nstk) - onesd * onesd' / nstk) * x_tplus1';

    if norm(x_tplus1_cent) ~= 0
        b_current = b_current + epsilon * x_tplus1_cent / norm(x_tplus1_cent);
    end

    b_next = simplex_projection_selfnorm2(b_current, 1);
