function export_olps_baselines_tail40_clipped(varargin)
%EXPORT_OLPS_BASELINES_TAIL40_CLIPPED  Re-run OLPS baselines on clipped tail40 data.
%
% This script applies relative-price clipping to each dataset, takes the
% tail test segment (default tail40 via dev_ratio=0.6), and re-runs the
% nine OLPS baselines used in the paper:
%   UBAH, BCRP, UP, OLMAR-2, RMR, Anticor, CORN, PPT, TPPT
%
% Outputs per-dataset OLPS-style .mat files into out_dir with names:
%   <algo>-<dataset>_tail40.mat
% containing at least: cum_ret, cumprod_ret, daily_ret.
%
% Usage:
%   export_olps_baselines_tail40_clipped('out_dir','results_xrelclip_p0p5_99p5');

    p = inputParser;
    addParameter(p, 'dev_ratio', 0.6);
    addParameter(p, 'tran_cost', 0.001);
    addParameter(p, 'data_dir', fullfile(fileparts(mfilename('fullpath')), 'Data Set'));
    addParameter(p, 'out_dir', fullfile(fileparts(mfilename('fullpath')), 'results_xrelclip'));
    addParameter(p, 'datasets', []); % [] for all, or e.g. {'inv500','msci'}
    addParameter(p, 'xrel_clip_mode', 'percentile'); % 'none' | 'fixed' | 'percentile'
    addParameter(p, 'xrel_clip_fixed', [0.5, 1.5]); % [lo, hi] when mode='fixed'
    addParameter(p, 'xrel_clip_prc', [0.5, 99.5]); % [p_lo, p_hi] when mode='percentile'
    parse(p, varargin{:});
    opts = p.Results;

    script_dir = fileparts(mfilename('fullpath'));
    out_dir = char(string(opts.out_dir));
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % Add baseline strategy paths.
    addpath('H:/OLPS-master/Strategy', '-begin');
    base_dir = fileparts(script_dir); % .../matlab code

    cleanupObj = onCleanup(@() cleanup_paths(base_dir)); %#ok<NASGU>

    files = dir(fullfile(opts.data_dir, '*.mat'));
    if isempty(files)
        error('No datasets found in %s', opts.data_dir);
    end
    [~, order] = sort({files.name});
    files = files(order);
    if ~isempty(opts.datasets)
        wanted = string(opts.datasets);
        wanted = lower(erase(wanted, ".mat"));
        keep = false(numel(files), 1);
        for ii = 1:numel(files)
            keep(ii) = any(lower(erase(string(files(ii).name), ".mat")) == wanted);
        end
        files = files(keep);
        if isempty(files)
            error('No datasets matched opts.datasets.');
        end
    end

    tc = double(opts.tran_cost);
    opts_env = struct();
    opts_env.quiet_mode = 1;
    opts_env.display_interval = 500;
    opts_env.progress = 0;
    opts_env.log_mode = 0;
    opts_env.mat_mode = 0;
    opts_env.analyze_mode = 0;
    opts_env.his = 0;

    fid = fopen(fullfile(out_dir, 'tmp_log.txt'), 'w');
    if fid == -1
        error('Cannot open baseline log file under out_dir=%s', out_dir);
    end
    fid_cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    xrel_clip_mode = lower(string(opts.xrel_clip_mode));

    for i = 1:numel(files)
        dataset = erase(files(i).name, '.mat');
        data_path = fullfile(files(i).folder, files(i).name);
        S = load(data_path, 'data');
        data = S.data;
        if xrel_clip_mode ~= "none"
            data = clip_xrel_local(data, xrel_clip_mode, opts.xrel_clip_fixed, opts.xrel_clip_prc);
        end

        T = size(data, 1);
        start_idx = floor(double(opts.dev_ratio) * T) + 1;
        data_tail = data(start_idx:end, :);

        fprintf('\n=== Baselines (clipped tail40): %s, tail=%d:%d (T=%d) ===\n', dataset, start_idx, T, T);

        % UBAH
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = ubah_run(fid, data_tail, tc, opts_env);
        save(fullfile(out_dir, sprintf('ubah-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        % BCRP
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = bcrp_run(fid, data_tail, tc, opts_env);
        save(fullfile(out_dir, sprintf('bcrp-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        % UP
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = up_run(fid, data_tail, tc, opts_env);
        save(fullfile(out_dir, sprintf('up-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        % OLMAR-2 (epsilon=10, alpha=0.5)
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = olmar2_run(fid, data_tail, 10, 0.5, tc, opts_env);
        save(fullfile(out_dir, sprintf('olmar2-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        % Anticor (W=30)
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = anticor_run(fid, data_tail, 30, tc, opts_env);
        save(fullfile(out_dir, sprintf('anticor-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        % CORN (w=5, c=0.1)
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = corn_run(fid, data_tail, 5, 0.1, tc, opts_env);
        save(fullfile(out_dir, sprintf('corn-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        % RMR (epsilon=0.5, W=5) - run signature differs; no daily_portfolio.
        [run_ret, total_ret, day_ret] = rmr_run(fid, data_tail, 0.5, tc, 5, opts_env); %#ok<ASGLU>
        cumprod_ret = total_ret(:);
        daily_ret = day_ret(:);
        cum_ret = cumprod_ret(end);
        save(fullfile(out_dir, sprintf('rmr-%s_tail40.mat', dataset)), 'cum_ret', 'cumprod_ret', 'daily_ret');

        % PPT / TPPT (use local implementations in ./PPT and ./TPPT)
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = run_ppt_like('PPT', data_tail, 5, 100, tc);
        save(fullfile(out_dir, sprintf('ppt-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
        [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = run_ppt_like('TPPT', data_tail, 5, 100, tc);
        save(fullfile(out_dir, sprintf('tppt-%s_tail40.mat', dataset)), ...
            'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
    end
end

function [cum_ret, cumprod_ret, daily_ret, daily_portfolio] = run_ppt_like(model_kind, data, win_size, epsilon, tran_cost)
    % This mirrors export_olps_mat_results_all's PPT/TPPT runner but operates on the given data slice.
    base_dir = fileparts(fileparts(mfilename('fullpath'))); % .../matlab code
    model_dir = fullfile(base_dir, upper(model_kind));
    addpath(model_dir, '-begin');
    cleanupObj = onCleanup(@() rmpath(model_dir)); %#ok<NASGU>
    clear PPT PPT_run simplex_projection_selfnorm2

    [T, N] = size(data);
    close_price = ones(T, N);
    for t = 2:T
        close_price(t, :) = close_price(t - 1, :) .* data(t, :);
    end

    daily_port = ones(N, 1) / N;
    daily_port_o = zeros(N, 1);
    daily_ret = ones(T, 1);
    daily_portfolio = zeros(T, N);

    for t = 1:T
        daily_portfolio(t, :) = daily_port';
        turnover_t = sum(abs(daily_port - daily_port_o));
        daily_ret(t, 1) = (data(t, :) * daily_port) * (1 - tran_cost / 2 * turnover_t);

        daily_port_o = daily_port .* data(t, :)' / (data(t, :) * daily_port);
        if t < T
            [daily_port_n, ~, ~] = PPT(close_price, data, t, daily_port, win_size, epsilon);
            daily_port = daily_port_n;
        end
    end

    cumprod_ret = cumprod(daily_ret);
    cum_ret = cumprod_ret(end);
end

function x_rel = clip_xrel_local(x_rel, mode, fixed_bounds, prc_bounds)
    mode = lower(string(mode));
    if mode == "none"
        return;
    end
    x = x_rel(:);
    if any(~isfinite(x))
        error('x_rel contains non-finite values, cannot clip safely.');
    end
    if mode == "fixed"
        lo = fixed_bounds(1);
        hi = fixed_bounds(2);
    elseif mode == "percentile"
        lo = prctile(x, prc_bounds(1));
        hi = prctile(x, prc_bounds(2));
    else
        error('Unsupported xrel clip mode: %s', mode);
    end
    if ~(isfinite(lo) && isfinite(hi) && lo > 0 && hi > lo)
        error('Invalid clip bounds computed (lo=%.6g, hi=%.6g).', lo, hi);
    end
    x_rel(x_rel < lo) = lo;
    x_rel(x_rel > hi) = hi;
end

function cleanup_paths(base_dir)
    try, rmpath('H:/OLPS-master/Strategy'); end %#ok<TRYNC>
    try, rmpath(fullfile(base_dir, 'PPT')); end %#ok<TRYNC>
    try, rmpath(fullfile(base_dir, 'TPPT')); end %#ok<TRYNC>
end
