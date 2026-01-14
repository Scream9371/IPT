function split = ipt_dev_test_split(T, varargin)
% ipt_dev_test_split - Time-based development/test split indices by ratio.
%
% Default: dev/test = 0.6/0.4.
%
% Outputs:
%   split.dev_start, split.dev_end, split.test_start, split.test_end

    p = inputParser;
    addParameter(p, 'dev_ratio', 0.6);
    parse(p, varargin{:});
    opts = p.Results;

    if T < 3
        error('T too small for split: T=%d', T);
    end
    if opts.dev_ratio <= 0 || opts.dev_ratio >= 1
        error('Invalid dev_ratio: %.4f', opts.dev_ratio);
    end

    dev_end = floor(T * opts.dev_ratio);
    test_start = dev_end + 1;
    if dev_end < 1 || test_start > T
        error('Invalid split for T=%d (dev_end=%d, test_start=%d)', T, dev_end, test_start);
    end

    split = struct();
    split.dev_start = 1;
    split.dev_end = dev_end;
    split.test_start = test_start;
    split.test_end = T;
    split.dev_ratio = opts.dev_ratio;
    split.test_ratio = 1 - opts.dev_ratio;
end

