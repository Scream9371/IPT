function smoothed = ipt_smooth_series(values, alpha)
% ipt_smooth_series - Exponential moving average smoothing.

    values = values(:);
    if isempty(values)
        smoothed = values;
        return;
    end
    smoothed = zeros(size(values));
    smoothed(1) = values(1);
    for i = 2:numel(values)
        smoothed(i) = alpha * values(i) + (1 - alpha) * smoothed(i - 1);
    end
end

