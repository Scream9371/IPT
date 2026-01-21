function [b_next, debug_stats] = IPT(p_close, x_rel, current_t, b_current, win_size, w_YAR, Q_factor, epsilon, force_no_orth, couple_mode, couple_param)
    % IPT - Investment Potential Tracking algorithm for portfolio selection
    %
    % Paper-aligned core update:
    %   s_hat = r_hat - Q*w
    %   b_next = Proj_{simplex}( b_t + epsilon * C*s_hat / ||C*s_hat|| )
    %
    % Optional orthogonalization (post-paper enhancement, controlled by force_no_orth):
    %   Remove the component of e_c that directly opposes the trend direction r_c.
    %   This keeps risk control but avoids systematically "killing" trend following.

    if nargin < 8 || isempty(epsilon)
        epsilon = 100;
    end

    if nargin < 9 || isempty(force_no_orth)
        force_no_orth = false;
    end

    if nargin < 10 || isempty(couple_mode)
        couple_mode = 0;
    end

    if nargin < 11 || isempty(couple_param)
        couple_param = 1;
    end

    nstk = size(x_rel, 2);

    if current_t < win_size + 1
        r_hat = x_rel(current_t, :);
    else
        closebefore = p_close((current_t - win_size + 1):(current_t), :);
        closepredict = max(closebefore);
        r_hat = closepredict ./ p_close(current_t, :);
    end

    onesd = ones(nstk, 1);
    C = eye(nstk) - (onesd * onesd') / nstk;
    r_c = C * r_hat';
    rc2 = dot(r_c, r_c);

    w_use = w_YAR(current_t, :);
    if couple_mode == 5
        w_ref = ones(1, nstk) / nstk;
        w_use = w_use - w_ref;
    elseif couple_mode == 6
        lambda = couple_param;
        proj_w = (w_use * r_c) / (rc2 + 1e-12);
        w_ref = lambda * proj_w * r_c';
        w_use = w_use - w_ref;
    end

    e_hat = Q_factor(current_t) .* w_use;
    e_c = C * e_hat';
    proj = 0;
    orth_applied = false;
    nr = norm(r_c);
    ne = norm(e_c);

    if ~force_no_orth && rc2 > 1e-12
        proj = dot(e_c, r_c) / rc2;
        % If proj > 0, -e_c reduces magnitude of r_c (opposes trend).
        % Strip only that conflicting component to preserve trend tracking.
        if proj > 0
            if couple_mode == 2
                lambda = max(0, min(1, couple_param));
                if lambda > 0
                    e_c = e_c - lambda * proj * r_c;
                    orth_applied = true;
                end
            else
                e_c = e_c - proj * r_c;
                orth_applied = true;
            end
        end
    elseif rc2 > 1e-12
        proj = dot(e_c, r_c) / rc2;
    end

    if couple_mode == 1
        gamma = max(0, couple_param);
        scale = min(1, gamma * norm(r_c) / (norm(e_c) + 1e-12));
        e_c = scale * e_c;
    elseif couple_mode == 4
        if numel(couple_param) >= 2
            gamma_risk = max(0, couple_param(1));
            gamma_rev = max(0, couple_param(2));
        else
            gamma_risk = max(0, couple_param);
            gamma_rev = gamma_risk;
        end

        if Q_factor(current_t) > 0
            gamma = gamma_risk;
        elseif Q_factor(current_t) < 0
            gamma = gamma_rev;
        else
            gamma = 0;
        end

        if gamma > 0
            scale = min(1, gamma * norm(r_c) / (norm(e_c) + 1e-12));
            e_c = scale * e_c;
        else
            e_c = 0 * e_c;
        end
    end

    x_tplus1_cent = (r_c - e_c);
    x_norm = norm(x_tplus1_cent);

    debug_stats.rc2 = rc2;
    debug_stats.proj = proj;
    debug_stats.orth_applied = orth_applied;
    debug_stats.nr = nr;
    debug_stats.ne = ne;
    debug_stats.x_norm = x_norm;

    eps_used = epsilon;
    if couple_mode == 3
        kappa = max(0, couple_param);
        eps_used = epsilon / (1 + kappa * abs(Q_factor(current_t)));
    end

    debug_stats.epsilon_eff = eps_used;

    if x_norm ~= 0
        b_current = b_current + eps_used * x_tplus1_cent / x_norm;
    end

    b_next = simplex_projection_selfnorm2(b_current, 1);
end
