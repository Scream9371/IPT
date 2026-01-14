function [w_YAR, Q_factor, state_meta] = active_function(yar_weights_long, yar_weights_near, yar_ubah_long, yar_ubah_near, data, win_long, reverse_factor, risk_factor, q_value, L_long_history, L_near_history, high_confirm_days, extreme_confirm_days)
    % active_function - Three-state selection strategy for IPT model portfolio adjustment
    %
    %   [w_YAR, Q, state_meta] = active_function(...)
    %   implements a three-state model selection strategy that adapts to varying
    %   market conditions by integrating recent trend, reversal potential and
    %   historical risk to calculate investment potential scores and model parameters.

    if nargin < 11 || isempty(L_near_history)
        L_near_history = L_long_history;
    end
    if nargin < 12 || isempty(high_confirm_days)
        high_confirm_days = 1;
    end
    if nargin < 13 || isempty(extreme_confirm_days)
        extreme_confirm_days = 3;
    end
    high_confirm_days = max(1, floor(double(high_confirm_days)));
    extreme_confirm_days = max(1, floor(double(extreme_confirm_days)));

    q = q_value;
    [datasets_T, datasets_N] = size(data);
    w_YAR = zeros(datasets_T, datasets_N);
    Q_factor = zeros(datasets_T, 1);
    cnt_high = 0;
    cnt_ext = 0;

    % Precompute UBAH return/price for downside gating (no extra hyperparameters).
    ubah_rel = mean(data, 2);
    ubah_price = cumprod(ubah_rel);
    near_win = max(2, floor(win_long / 2));
    near_return_raw = nan(datasets_T, 1);
    near_drawdown_raw = nan(datasets_T, 1);
    for t = 1:datasets_T
        if t >= near_win
            idx = (t - near_win + 1):t;
            near_return_raw(t) = prod(ubah_rel(idx)) - 1;
            peak = max(ubah_price(idx));
            near_drawdown_raw(t) = max(0, peak - ubah_price(t));
        end
    end

    % Diagnostics aligned to Q_factor timeline.
    state_code = nan(datasets_T, 1); % 1/2: reverse, 3: normal, 4: high-risk, 5: extreme-risk
    downside_cond = false(datasets_T, 1);
    near_return = nan(datasets_T, 1);
    near_drawdown = nan(datasets_T, 1);
    risk_active = false(datasets_T, 1);
    L_used = nan(datasets_T, 1);
    yar_ubah_long_used = nan(datasets_T, 1);
    yar_ubah_near_used = nan(datasets_T, 1);

    for i = 1:datasets_T - win_long

        if isempty(L_long_history)
            L_long = 0;
        elseif i <= numel(L_long_history)
            L_long = L_long_history(i);
        else
            L_long = L_long_history(end);
        end

        if yar_ubah_long(i) <= q * L_long / 2
            Q_factor(i + win_long) = -2 * reverse_factor;
            w_YAR(i + win_long, :) = yar_weights_long(i, :);
            cnt_high = 0;
            cnt_ext = 0;
            state_code(i + win_long) = 1;
        elseif yar_ubah_long(i) <= q * L_long
            Q_factor(i + win_long) = -reverse_factor;
            w_YAR(i + win_long, :) = yar_weights_long(i, :);
            cnt_high = 0;
            cnt_ext = 0;
            state_code(i + win_long) = 2;
        else
            near_index = i + floor(win_long / 2);
            if near_index <= size(yar_ubah_near, 1)
                if isempty(L_near_history)
                    L_near = 0;
                elseif near_index <= numel(L_near_history)
                    L_near = L_near_history(near_index);
                else
                    L_near = L_near_history(end);
                end

                thr0 = (1 - q) * L_near;
                thr1 = (1 - q / 2) * L_near;
                yN = yar_ubah_near(near_index);

                % Downside gate: only treat Q>0 as risk when downside is present.
                near_return(i + win_long) = near_return_raw(near_index);
                near_drawdown(i + win_long) = near_drawdown_raw(near_index);
                downside_ok = false;
                if isfinite(near_return(i + win_long)) && isfinite(near_drawdown(i + win_long))
                    downside_ok = (near_return(i + win_long) < 0) || (near_drawdown(i + win_long) > 0);
                end
                downside_cond(i + win_long) = downside_ok;

                if ~downside_ok
                    cnt_high = 0;
                    cnt_ext = 0;
                    Q_factor(i + win_long) = 0;
                    w_YAR(i + win_long, :) = yar_weights_near(near_index, :);
                    state_code(i + win_long) = 3;
                elseif yN <= thr0
                    cnt_high = 0;
                    cnt_ext = 0;
                    Q_factor(i + win_long) = 0;
                    w_YAR(i + win_long, :) = yar_weights_near(near_index, :);
                    state_code(i + win_long) = 3;
                elseif yN <= thr1
                    cnt_high = cnt_high + 1;
                    cnt_ext = 0;
                    if cnt_high >= high_confirm_days
                        Q_factor(i + win_long) = risk_factor;
                        state_code(i + win_long) = 4;
                    else
                        Q_factor(i + win_long) = 0;
                        state_code(i + win_long) = 3;
                    end
                    w_YAR(i + win_long, :) = yar_weights_near(near_index, :);
                else
                    cnt_high = cnt_high + 1;
                    cnt_ext = cnt_ext + 1;
                    if cnt_ext >= extreme_confirm_days
                        Q_factor(i + win_long) = 2 * risk_factor;
                        state_code(i + win_long) = 5;
                    elseif cnt_high >= high_confirm_days
                        Q_factor(i + win_long) = risk_factor;
                        state_code(i + win_long) = 4;
                    else
                        Q_factor(i + win_long) = 0;
                        state_code(i + win_long) = 3;
                    end
                    w_YAR(i + win_long, :) = yar_weights_near(near_index, :);
                end
                yar_ubah_near_used(i + win_long) = yN;
            else
                % Default behavior if near_index is out of bounds
                Q_factor(i + win_long) = 0;
                w_YAR(i + win_long, :) = yar_weights_long(i, :);
                cnt_high = 0;
                cnt_ext = 0;
                state_code(i + win_long) = 3;
            end
        end
        L_used(i + win_long) = L_long;
        yar_ubah_long_used(i + win_long) = yar_ubah_long(i);
        risk_active(i + win_long) = Q_factor(i + win_long) > 0;
    end

    state_meta = struct();
    state_meta.state_code = state_code;
    state_meta.downside_condition = downside_cond;
    state_meta.near_return = near_return;
    state_meta.near_drawdown = near_drawdown;
    state_meta.risk_active = risk_active;
    state_meta.L = L_used;
    state_meta.yar_ubah_long = yar_ubah_long_used;
    state_meta.yar_ubah_near = yar_ubah_near_used;
end
