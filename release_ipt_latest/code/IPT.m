function [b_next] = IPT(p_close, x_rel, current_t, b_current, win_size, w_YAR, Q_factor, epsilon, force_no_orth)
    % IPT - Investment Potential Tracking algorithm for portfolio selection
    %
    % Inputs:
    %   epsilon       - Step size (default 100)
    %   force_no_orth - If true, skip orthogonalization step (default false)

    if nargin < 8 || isempty(epsilon)
        % Default to 100 if not provided, but allow external control
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

    % Conditional Orthogonalization (Risk Stripping)
    % Protect trend direction: remove risk component that opposes trend
    % Only performed if force_no_orth is false
    if ~force_no_orth
        rc2 = dot(r_c, r_c);

        if rc2 > 1e-12
            proj = dot(e_c, r_c) / rc2;
            % If proj > 0, -e_c reduces magnitude of r_c (opposes trend)
            if proj > 0
                e_c = e_c - proj * r_c;
            end

        end

    end

    % Scale in centered space (more consistent)
    scale = min(1, norm(r_c, 2) / (norm(e_c, 2) +1e-12));

    % Update direction directly in centered space
    % Result is already zero-mean, so explicit centering is not needed
    x_tplus1_cent = (r_c - scale * e_c);

    if norm(x_tplus1_cent) ~= 0
        b_current = b_current + epsilon * x_tplus1_cent / norm(x_tplus1_cent);
    end

    b_next = simplex_projection_selfnorm2(b_current, 1);
end
