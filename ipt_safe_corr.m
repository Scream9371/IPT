function r = ipt_safe_corr(a, b)
% ipt_safe_corr - Safe Pearson correlation for vectors.

    a = a(:);
    b = b(:);
    if numel(a) ~= numel(b) || isempty(a)
        r = nan;
        return;
    end
    if all(a == a(1)) || all(b == b(1))
        r = nan;
        return;
    end
    C = corrcoef(a, b);
    r = C(1, 2);
end

