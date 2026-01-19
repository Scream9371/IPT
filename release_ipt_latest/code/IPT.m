function [b_next, debug_stats] = IPT(p_close, x_rel, current_t, b_current, win_size, w_YAR, Q_factor, epsilon, force_no_orth)
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

    nstk = size(x_rel, 2);

    if current_t < win_size + 1
        r_hat = x_rel(current_t, :);
    else
        closebefore = p_close((current_t - win_size + 1):(current_t), :);
        closepredict = max(closebefore);
        r_hat = closepredict ./ p_close(current_t, :);
    end

    e_hat = Q_factor(current_t) .* w_YAR(current_t, :);

    onesd = ones(nstk, 1);
    C = eye(nstk) - (onesd * onesd') / nstk;
    r_c = C * r_hat';
    e_c = C * e_hat';

    rc2 = dot(r_c, r_c);
    proj = 0;
    orth_applied = false;

    if ~force_no_orth && rc2 > 1e-12
        proj = dot(e_c, r_c) / rc2;
        % If proj > 0, -e_c reduces magnitude of r_c (opposes trend).
        % Strip only that conflicting component to preserve trend tracking.
        if proj > 0
            e_c = e_c - proj * r_c;
            orth_applied = true;
        end
    elseif rc2 > 1e-12
        proj = dot(e_c, r_c) / rc2;
    end

    x_tplus1_cent = (r_c - e_c);

    debug_stats.rc2 = rc2;
    debug_stats.proj = proj;
    debug_stats.orth_applied = orth_applied;

    if norm(x_tplus1_cent) ~= 0
        b_current = b_current + epsilon * x_tplus1_cent / norm(x_tplus1_cent);
    end

    b_next = simplex_projection_selfnorm2(b_current, 1);
end
