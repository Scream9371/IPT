function [w_YAR, Q_factor, state_meta] = active_function(yar_weights_long, yar_weights_near, yar_ubah_long, yar_ubah_near, data, win_long, reverse_factor, risk_factor, q_value, L_long_history, L_near_history)

    q = q_value;
    [datasets_T, datasets_N] = size(data);
    w_YAR = zeros(datasets_T, datasets_N);
    Q_factor = zeros(datasets_T, 1);
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

        t = i + win_long;
        L_used(t) = L_long;

        if i <= size(yar_ubah_long, 1)
            yar_ubah_long_used(t) = yar_ubah_long(i);
        end

        if L_long <= 0
            continue;
        end

        near_index = i + floor(win_long / 2);
        yN = 0;

        if near_index <= size(yar_ubah_near, 1)
            yN = yar_ubah_near(near_index);
            yar_ubah_near_used(t) = yN;

            if near_index <= size(yar_weights_near, 1)
                w_YAR(t, :) = yar_weights_near(near_index, :);
            end

        else

            if i <= size(yar_ubah_long, 1)
                yN = yar_ubah_long(i);
                yar_ubah_near_used(t) = yN;
            end

            if i <= size(yar_weights_long, 1)
                w_YAR(t, :) = yar_weights_long(i, :);
            end

        end

        if ~isfinite(yN)
            continue;
        end

        base = q * L_long;
        span = (1 - q) * L_long;

        if span <= 0
            span = L_long;
        end

        denom = max(span, eps);
        stress = (yN - base) / denom;
        stress = max(0, min(1, stress));

        Q_factor(t) = risk_factor * stress;
    end

    state_meta = struct();
    state_meta.L = L_used;
    state_meta.yar_ubah_long = yar_ubah_long_used;
    state_meta.yar_ubah_near = yar_ubah_near_used;
end
