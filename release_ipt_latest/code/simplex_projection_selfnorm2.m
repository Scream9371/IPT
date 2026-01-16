function [b_next, rho] = simplex_projection_selfnorm2(b_current, simplex_size)
    % This function is the simplex projection function exploited by Peak Price Tracking (PPT)[1]
    % and Adaptive Input and Composite Trend Representation (AICTR)[2]. It originates from [4][5].
    %
    % Reference:
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
    % b_current          - a d-dimensional portfolio weight vector
    % simplex_size       - the "size" of the simplex, default=1
    %
    % Outputs:
    % b_next             - the projected weight vector satisfying constraints on the simplex

    while (max(abs(b_current)) > 1e6)
        b_current = b_current / 10;
    end

    % Sort components in descending order
    v_sorted = sort(b_current, 'descend');

    % Compute cumulative sums [4]
    sv = cumsum(v_sorted);

    % Find largest index ρ satisfying: u(ρ) > [∑_{i=1}^{ρ} u(i) - simplex_size]/ρ [4]
    rho = find(v_sorted > (sv - simplex_size) ./ (1:length(v_sorted))', 1, 'last');

    % Compute shrinkage threshold θ [4]
    theta = (sv(rho) - simplex_size) / rho;

    b_next = max(b_current - theta, 0);
end
