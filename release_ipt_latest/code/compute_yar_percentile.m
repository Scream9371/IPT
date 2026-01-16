function L_history = compute_yar_percentile(yar_ubah_series, percentile)
% compute_yar_percentile - Running percentile using past-only history.
%
%   L_history = compute_yar_percentile(yar_ubah_series, percentile)
%   computes a running percentile for a YAR series. L_history(i) uses
%   values up to t-1 to avoid look-ahead.

    if nargin < 2
        percentile = 95;
    end

    yar_ubah_series = yar_ubah_series(:);
    n = numel(yar_ubah_series);
    L_history = zeros(n, 1);

    for i = 1:n
        if i == 1
            sample = yar_ubah_series(1);
        else
            sample = yar_ubah_series(1:i-1);
        end
        sample = sample(isfinite(sample));
        if isempty(sample)
            sample = yar_ubah_series(max(1, i));
        end
        sample = sort(sample);
        idx = max(1, ceil(percentile / 100 * numel(sample)));
        L_history(i) = sample(idx);
    end
end
