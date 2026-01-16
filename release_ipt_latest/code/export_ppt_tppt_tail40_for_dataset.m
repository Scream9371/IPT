function export_ppt_tppt_tail40_for_dataset(dataset, varargin)
%EXPORT_PPT_TPPT_TAIL40_FOR_DATASET  Export PPT/TPPT tail40 .mat for one dataset.
%
% This helper is used to fill missing PPT/TPPT baseline results (e.g., MARPD)
% when other baselines already exist as *_tail40.mat files.
%
% Output files (OLPS-style minimal fields):
%   - ppt-<dataset>_tail40.mat   (cumprod_ret, daily_ret)
%   - tppt-<dataset>_tail40.mat  (cumprod_ret, daily_ret)
%
    p = inputParser;
    addParameter(p, 'data_dir', fullfile(fileparts(mfilename('fullpath')), 'Data Set'));
    addParameter(p, 'out_dir', fullfile(fileparts(mfilename('fullpath')), 'results'));
    addParameter(p, 'dev_ratio', 0.6);
    addParameter(p, 'win_size', 5);
    addParameter(p, 'epsilon', 100);
    addParameter(p, 'tran_cost', 0.001);
    parse(p, varargin{:});
    opts = p.Results;

    dataset = lower(string(dataset));
    if strlength(dataset) == 0
        error('dataset must be non-empty.');
    end

    data_path = fullfile(opts.data_dir, dataset + ".mat");
    if ~isfile(data_path)
        error('Dataset not found: %s', data_path);
    end
    if ~exist(opts.out_dir, 'dir')
        mkdir(opts.out_dir);
    end

    S = load(data_path, 'data');
    data = S.data;
    T = size(data, 1);
    start_idx = floor(double(opts.dev_ratio) * T) + 1;
    if start_idx > T
        error('Invalid dev_ratio=%.4g for T=%d.', opts.dev_ratio, T);
    end
    data_tail = data(start_idx:end, :);

    [cumprod_ret, daily_ret] = run_ppt_like_local('PPT', data_tail, opts.win_size, opts.epsilon, opts.tran_cost);
    save(fullfile(opts.out_dir, sprintf('ppt-%s_tail40.mat', dataset)), 'cumprod_ret', 'daily_ret');

    [cumprod_ret, daily_ret] = run_ppt_like_local('TPPT', data_tail, opts.win_size, opts.epsilon, opts.tran_cost);
    save(fullfile(opts.out_dir, sprintf('tppt-%s_tail40.mat', dataset)), 'cumprod_ret', 'daily_ret');
end

function [cumprod_ret, daily_ret] = run_ppt_like_local(model_kind, data, win_size, epsilon, tran_cost)
    base_dir = fileparts(fileparts(mfilename('fullpath'))); % .../matlab code
    model_dir = fullfile(base_dir, upper(string(model_kind)));
    addpath(model_dir, '-begin');
    cleanupObj = onCleanup(@() rmpath(model_dir)); %#ok<NASGU>
    clear PPT PPT_run simplex_projection_selfnorm2

    [T, N] = size(data);
    close_price = ones(T, N);
    for t = 2:T
        close_price(t, :) = close_price(t - 1, :) .* data(t, :);
    end

    b = ones(N, 1) / N;
    b0 = zeros(N, 1);
    daily_ret = ones(T, 1);
    for t = 1:T
        turnover_t = sum(abs(b - b0));
        daily_ret(t, 1) = (data(t, :) * b) * (1 - tran_cost / 2 * turnover_t);

        b0 = b .* data(t, :)' / (data(t, :) * b);
        if t < T
            [b_next, ~, ~] = PPT(close_price, data, t, b, win_size, epsilon);
            b = b_next;
        end
    end

    cumprod_ret = cumprod(daily_ret);
end

