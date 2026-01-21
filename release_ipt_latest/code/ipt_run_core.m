function [cum_wealth, daily_incre_fact, b_history, debug_info] = ipt_run_core(x_rel, win_size, trans_cost, w_YAR, Q_factor, epsilon, varargin)
    % ipt_run_core - Unified core execution loop for IPT strategy.
    %
    % Inputs:
    %   x_rel            - Price relatives (daily returns + 1), T x N matrix
    %   win_size         - Lookback window size for peak price
    %   trans_cost       - Transaction cost (e.g., 0.001)
    %   w_YAR            - Weights from Active Function (T x N)
    %   Q_factor         - Q factor from Active Function (T x 1)
    %   epsilon          - Step size for IPT update (default 100)
    %   update_mix         - Mixing factor for new portfolio (default 1.0 = no inertia)
    %   max_turnover       - Maximum allowed daily turnover (default Inf)
    %   adaptive_inertia_q - (Optional) Flag or threshold for Q-based inertia
    %   force_no_orth      - (Optional) Disable orth strip
    %   condmix_mode       - (Optional) Conditional mixing mode (0=off,1=switch-fast)
    %   couple_mode        - (Optional) e_hat coupling mode
    %   couple_param       - (Optional) coupling parameter
    %
    % Outputs:
    %   cum_wealth       - Cumulative wealth vector (T x 1)
    %   daily_incre_fact - Daily returns (T x 1)
    %   b_history        - Portfolio weights history (N x T)
    %   debug_info       - Struct with diagnostic stats (proj, rc2)

    if nargin < 6 || isempty(epsilon), epsilon = 100; end

    update_mix = 1.0;
    max_turnover = Inf;
    adaptive_inertia_q = 0;
    force_no_orth = false;
    condmix_mode = 1;
    couple_mode = 0;
    couple_param = 1;

    if ~isempty(varargin)
        if numel(varargin) >= 1 && ~isempty(varargin{1}), update_mix = varargin{1}; end
        if numel(varargin) >= 2 && ~isempty(varargin{2}), max_turnover = varargin{2}; end
        if numel(varargin) >= 3 && ~isempty(varargin{3}), adaptive_inertia_q = varargin{3}; end
        if numel(varargin) >= 4 && ~isempty(varargin{4}), force_no_orth = varargin{4}; end
        if numel(varargin) >= 5 && ~isempty(varargin{5}), condmix_mode = varargin{5}; end
        if numel(varargin) >= 6 && ~isempty(varargin{6}), couple_mode = varargin{6}; end
        if numel(varargin) >= 7 && ~isempty(varargin{7}), couple_param = varargin{7}; end
    end

    [T, N] = size(x_rel);
    cum_wealth = ones(T, 1);
    daily_incre_fact = ones(T, 1);
    
    b_current = ones(N, 1) / N;
    b_history = zeros(N, T); % Corrected orientation to N x T to match IPT_run convention
    b_prev = zeros(N, 1);
    
    % Diagnostic arrays
    hist_proj = zeros(T, 1);
    hist_rc2 = zeros(T, 1);
    hist_orth = false(T, 1);
    hist_turnover = zeros(T, 1);
    hist_nr = zeros(T, 1);
    hist_ne = zeros(T, 1);
    hist_xnorm = zeros(T, 1);
    hist_eps = zeros(T, 1);
    
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
        hist_turnover(t) = turnover;
        daily_incre_fact(t, 1) = (x_rel(t, :) * b_current) * (1 - trans_cost / 2 * turnover);
        
        run_ret = run_ret * daily_incre_fact(t, 1);
        cum_wealth(t) = run_ret;
        
        % Portfolio evolution due to price changes
        b_prev_evolved = b_current .* x_rel(t, :)' / (x_rel(t, :) * b_current);
        b_prev = b_prev_evolved;
        
        if t < T
            % Get target portfolio from IPT algo
            % Note: IPT.m now accepts epsilon
            [b_target, step_stats] = IPT(p_close, x_rel, t, b_current, win_size, w_YAR, Q_factor, epsilon, force_no_orth, couple_mode, couple_param);
            
            hist_proj(t) = step_stats.proj;
            hist_rc2(t) = step_stats.rc2;
            if isfield(step_stats, 'orth_applied')
                hist_orth(t) = step_stats.orth_applied;
            end
            if isfield(step_stats, 'nr')
                hist_nr(t) = step_stats.nr;
            end
            if isfield(step_stats, 'ne')
                hist_ne(t) = step_stats.ne;
            end
            if isfield(step_stats, 'x_norm')
                hist_xnorm(t) = step_stats.x_norm;
            end
            if isfield(step_stats, 'epsilon_eff')
                hist_eps(t) = step_stats.epsilon_eff;
            end

            % Apply Mixing (Inertia)
            % condmix_mode=0: always use update_mix
            % condmix_mode=1: react fast only on state switch into non-neutral
            alpha = update_mix;
            if condmix_mode == 1
                if abs(Q_factor(t)) >= 1e-6
                    if t > 1 && abs(Q_factor(t) - Q_factor(t - 1)) >= 1e-6
                        alpha = 1.0;
                    end
                end
            end
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
    
    debug_info.proj = hist_proj;
    debug_info.rc2 = hist_rc2;
    debug_info.orth_applied = hist_orth;
    debug_info.turnover = hist_turnover;
    debug_info.nr = hist_nr;
    debug_info.ne = hist_ne;
    debug_info.x_norm = hist_xnorm;
    debug_info.epsilon_eff = hist_eps;
end
