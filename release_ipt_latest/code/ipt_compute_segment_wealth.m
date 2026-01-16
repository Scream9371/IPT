function [wealth_seg, wealth_before] = ipt_compute_segment_wealth(daily_incre, segment)
% ipt_compute_segment_wealth - Segment-reset wealth series (start at 1 for each segment).
% Returns:
%   wealth_seg    - cumprod of daily_incre within each segment
%   wealth_before - wealth at beginning of each day (1 for segment start)

    T = numel(daily_incre);
    wealth_seg = nan(T, 1);
    wealth_before = nan(T, 1);

    seg = string(segment(:));
    unique_segs = unique(seg, 'stable');
    for i = 1:numel(unique_segs)
        s = unique_segs(i);
        idx = find(seg == s);
        if isempty(idx)
            continue;
        end
        start_idx = idx(1);
        end_idx = idx(end);

        wealth_seg(start_idx:end_idx) = cumprod(daily_incre(start_idx:end_idx));
        wealth_before(start_idx) = 1;
        if end_idx > start_idx
            wealth_before((start_idx + 1):end_idx) = wealth_seg(start_idx:(end_idx - 1));
        end
    end
end

