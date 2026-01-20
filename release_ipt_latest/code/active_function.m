function [w_YAR, Q_factor] = active_function(yar_weights_long, yar_weights_near, yar_ubah_long, yar_ubah_near, data, win_long, risk_factor, q_value, L_long_history, varargin)
    %ACTIVE_FUNCTION Three-state switching rule in Algorithm 1 (paper.tex).
    %
    % Outputs
    %   Q_factor(t) in { -beta_reverse*alpha_reverse, -alpha_reverse, 0, alpha_risk, beta_risk*alpha_risk }
    %   w_YAR(t,:) selects long-term or near-term YAR weights accordingly.
    %
    % Backward compatible signature:
    %   active_function(..., risk_factor, q_value, L_long_history)
    % This variant uses rolling quantile thresholds on y_L and y_N
    % with a fixed window length (default 252), and ignores L_long_history.
    %
    % Optional overrides via varargin:
    %   'reverse_factor' (default = risk_factor)
    %   'beta_reverse'   (default = 2)
    %   'beta_risk'      (default = 2)

    reverse_factor = risk_factor;
    beta_reverse = 2;
    beta_risk = 2;
    debug_qzero = false;
    quantile_window = 252;
    min_hist = 30;

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
            elseif k == "debug_qzero"
                debug_qzero = logical(v);
            end
        end
    end

    q = double(q_value);
    [datasets_T, datasets_N] = size(data);
    w_YAR = zeros(datasets_T, datasets_N);
    Q_factor = zeros(datasets_T, 1);

    if q <= 0
        if debug_qzero
            fprintf('[active_function] q<=0 -> Q_factor all zeros (T=%d)\n', datasets_T);
        end
        return;
    end

    for i = 1:(datasets_T - win_long)
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

        hist_end = t - 1;
        hist_start = max(1, hist_end - quantile_window + 1);
        if hist_end < hist_start || hist_end < 1
            continue;
        end

        yL_hist = yar_ubah_long(hist_start:hist_end, 1);
        yN_hist = yar_ubah_near(hist_start:hist_end, 1);
        yL_hist = yL_hist(isfinite(yL_hist));
        yN_hist = yN_hist(isfinite(yN_hist));

        if numel(yL_hist) < min_hist || numel(yN_hist) < min_hist
            continue;
        end

        TL1 = prctile(yL_hist, 100 * (q / 2));
        TL2 = prctile(yL_hist, 100 * q);
        TN1 = prctile(yN_hist, 100 * (1 - q));
        TN2 = prctile(yN_hist, 100 * (1 - q / 2));

        % Algorithm 1: reversal states decided by long-term UBAH YAR
        if yL <= TL1
            Q_factor(t) = -beta_reverse * reverse_factor;
            w_YAR(t, :) = w_long;
        elseif yL <= TL2
            Q_factor(t) = -reverse_factor;
            w_YAR(t, :) = w_long;
        else
            % Algorithm 1: momentum/risk states decided by near-term UBAH YAR
            if yN <= TN1
                Q_factor(t) = 0;
                w_YAR(t, :) = w_near;
            elseif yN <= TN2
                Q_factor(t) = risk_factor;
                w_YAR(t, :) = w_near;
            else
                Q_factor(t) = beta_risk * risk_factor;
                w_YAR(t, :) = w_near;
            end
        end

    end

end
