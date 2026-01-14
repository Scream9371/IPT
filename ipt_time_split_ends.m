function split = ipt_time_split_ends(T, varargin)
% ipt_time_split_ends - Time-based train/val/test split indices by ratio.
%
% Default: 6/2/2 split (train=0.6, val=0.2, test=0.2).
%
% Outputs:
%   split.train_end, split.val_start, split.val_end, split.test_start, split.test_end

    p = inputParser;
    addParameter(p, 'train_ratio', 0.6);
    addParameter(p, 'val_ratio', 0.2);
    parse(p, varargin{:});
    opts = p.Results;

    if T < 3
        error('T too small for split: T=%d', T);
    end
    if opts.train_ratio <= 0 || opts.val_ratio <= 0 || opts.train_ratio + opts.val_ratio >= 1
        error('Invalid ratios: train_ratio=%.4f, val_ratio=%.4f', opts.train_ratio, opts.val_ratio);
    end

    train_end = floor(T * opts.train_ratio);
    val_end = floor(T * (opts.train_ratio + opts.val_ratio));
    val_start = train_end + 1;
    test_start = val_end + 1;

    if train_end < 1 || val_start > val_end || test_start > T
        error('Invalid split for T=%d (train_end=%d, val=%d:%d, test_start=%d)', ...
            T, train_end, val_start, val_end, test_start);
    end

    split = struct();
    split.train_end = train_end;
    split.val_start = val_start;
    split.val_end = val_end;
    split.test_start = test_start;
    split.test_end = T;
    split.train_ratio = opts.train_ratio;
    split.val_ratio = opts.val_ratio;
    split.test_ratio = 1 - opts.train_ratio - opts.val_ratio;
end

