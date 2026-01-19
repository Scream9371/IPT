function [w_YAR, Q_factor] = active_function(yar_weights_long, yar_weights_near, yar_ubah_long, yar_ubah_near, data, win_long, risk_factor, q_value, L_long_history, varargin)
    %ACTIVE_FUNCTION Three-state switching rule in Algorithm 1 (paper.tex).
    %
    % Outputs
    %   Q_factor(t) in { -beta_reverse*alpha_reverse, -alpha_reverse, 0, alpha_risk, beta_risk*alpha_risk }
    %   w_YAR(t,:) selects long-term or near-term YAR weights accordingly.
    %
    % Backward compatible signature:
    %   active_function(..., risk_factor, q_value, L_long_history)
    % Optional overrides via varargin:
    %   'reverse_factor' (default = risk_factor)
    %   'beta_reverse'   (default = 2)
    %   'beta_risk'      (default = 2)

    reverse_factor = risk_factor;
    beta_reverse = 2;
    beta_risk = 2;

    if ~isempty(varargin)
        for i = 1:2:numel(varargin)
            if i + 1 > numel(varargin), break; end
            k = lower(string(varargin{i}));
            v = varargin{i + 1};

            if k == "reverse_factor"
                reverse_factor = double(v);
            elseif k == "beta_reverse"
                beta_reverse = double(v);
            elseif k == "beta_risk"
                beta_risk = double(v);
            end
        end
    end

    q = double(q_value);
    [datasets_T, datasets_N] = size(data);
    w_YAR = zeros(datasets_T, datasets_N);
    Q_factor = zeros(datasets_T, 1);

    for i = 1:(datasets_T - win_long)
        if isempty(L_long_history)
            L = 0;
        elseif i <= numel(L_long_history)
            L = L_long_history(i);
        else
            L = L_long_history(end);
        end

        if ~(isfinite(L) && L > 0)
            continue;
        end

        t = i + win_long;

        yL = NaN;
        if i <= size(yar_ubah_long, 1)
            yL = yar_ubah_long(i);
        end

        near_index = i + floor(win_long / 2);
        yN = NaN;

        if near_index <= size(yar_ubah_near, 1)
            yN = yar_ubah_near(near_index);
        end

        w_long = zeros(1, datasets_N);
        if i <= size(yar_weights_long, 1)
            w_long = yar_weights_long(i, :);
        end

        w_near = w_long;
        if near_index <= size(yar_weights_near, 1)
            w_near = yar_weights_near(near_index, :);
        end

        if ~(isfinite(yL) && isfinite(yN))
            continue;
        end

        % Algorithm 1: reversal states decided by long-term UBAH YAR
        if yL <= (q * L) / 2
            Q_factor(t) = -beta_reverse * reverse_factor;
            w_YAR(t, :) = w_long;
        elseif yL <= (q * L)
            Q_factor(t) = -reverse_factor;
            w_YAR(t, :) = w_long;
        else
            % Algorithm 1: momentum/risk states decided by near-term UBAH YAR
            if yN <= (1 - q) * L
                Q_factor(t) = 0;
                w_YAR(t, :) = w_near;
            elseif yN <= (1 - q / 2) * L
                Q_factor(t) = risk_factor;
                w_YAR(t, :) = w_near;
            else
                Q_factor(t) = beta_risk * risk_factor;
                w_YAR(t, :) = w_near;
            end
        end

    end

end
