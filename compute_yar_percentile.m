function L_history = compute_yar_percentile(yar_ubah_series, percentile)
% compute_yar_percentile - Running percentile for UBAH YAR history.
%
%   L_history = compute_yar_percentile(yar_ubah_series, percentile)
%   computes the running percentile (nearest-rank) for a YAR series.
%
% Inputs:
%   yar_ubah_series  - n x 1 (or 1 x n) vector of UBAH YAR values
%   percentile       - scalar in [0, 100], e.g., 95
%
% Output:
%   L_history        - n x 1 vector, L_history(i) is the percentile of
%                      yar_ubah_series(1:i)

    if nargin < 2
        percentile = 95;
    end

    yar_ubah_series = yar_ubah_series(:);
    n = numel(yar_ubah_series);
    L_history = zeros(n, 1);

    for i = 1:n
        sample = sort(yar_ubah_series(1:i));
        idx = max(1, ceil(percentile / 100 * i));
        L_history(i) = sample(idx);
    end
end
