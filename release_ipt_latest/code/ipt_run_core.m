function [cum_wealth, daily_incre_fact, b_history] = ipt_run_core(x_rel, win_size, trans_cost, w_YAR, Q_factor, epsilon, update_mix, max_turnover, adaptive_inertia_q, force_no_orth)
    % ipt_run_core - Unified core execution loop for IPT strategy.
    %
    % Inputs:
    %   x_rel            - Price relatives (daily returns + 1), T x N matrix
    %   win_size         - Lookback window size for peak price
    %   trans_cost       - Transaction cost (e.g., 0.001)
    %   w_YAR            - Weights from Active Function (T x N)
    %   Q_factor         - Q factor from Active Function (T x 1)
    %   epsilon          - Step size for IPT update (default 100)
    %   update_mix       - Mixing factor for new portfolio (default 1.0 = no inertia)
    %   max_turnover     - Maximum allowed daily turnover (default Inf)
    %   adaptive_inertia_q - (Optional) Flag or threshold for Q-based inertia
    %
    % Outputs:
    %   cum_wealth       - Cumulative wealth vector (T x 1)
    %   daily_incre_fact - Daily returns (T x 1)
    %   b_history        - Portfolio weights history (N x T)

    if nargin < 6 || isempty(epsilon), epsilon = 100; end
    if nargin < 7 || isempty(update_mix), update_mix = 1.0; end
    if nargin < 8 || isempty(max_turnover), max_turnover = Inf; end
    if nargin < 9 || isempty(adaptive_inertia_q), adaptive_inertia_q = 0; end
    if nargin < 10 || isempty(force_no_orth), force_no_orth = false; end

    [T, N] = size(x_rel);
    cum_wealth = ones(T, 1);
    daily_incre_fact = ones(T, 1);
    
    b_current = ones(N, 1) / N;
    b_history = zeros(N, T); % Corrected orientation to N x T to match IPT_run convention
    b_prev = zeros(N, 1);
    
    % Reconstruct price series (needed for IPT core)
    p_close = ones(T, N);
    for i = 2:T
        p_close(i, :) = p_close(i-1, :) .* x_rel(i, :);
    end
    
    run_ret = 1;
    
    for t = 1:T
        b_history(:, t) = b_current;
        
        % Calculate return with transaction cost
        turnover = sum(abs(b_current - b_prev));
        daily_incre_fact(t, 1) = (x_rel(t, :) * b_current) * (1 - trans_cost / 2 * turnover);
        
        run_ret = run_ret * daily_incre_fact(t, 1);
        cum_wealth(t) = run_ret;
        
        % Portfolio evolution due to price changes
        b_prev_evolved = b_current .* x_rel(t, :)' / (x_rel(t, :) * b_current);
        b_prev = b_prev_evolved;
        
        if t < T
            % Get target portfolio from IPT algo
            % Note: IPT.m now accepts epsilon
            b_target = IPT(p_close, x_rel, t, b_current, win_size, w_YAR, Q_factor, epsilon, force_no_orth);
            
            % Apply Mixing (Inertia)
            alpha = update_mix;
            if adaptive_inertia_q
                alpha = alpha * (1 / (1 + abs(Q_factor(t))));
            end
            alpha = max(0, min(1, alpha));
            
            % If alpha < 1, we keep some of the old portfolio
            if alpha < 1
                b_next = alpha * b_target + (1 - alpha) * b_current;
            else
                b_next = b_target;
            end
            
            % Apply Turnover Constraint
            if ~isinf(max_turnover)
                diff = b_next - b_prev_evolved;
                turnover_req = sum(abs(diff));
                if turnover_req > max_turnover
                    scale = max_turnover / turnover_req;
                    b_next = b_prev_evolved + diff * scale;
                end
            end
            
            b_current = b_next;
        end
    end
end
