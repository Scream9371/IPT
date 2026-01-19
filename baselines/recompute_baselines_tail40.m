function recompute_baselines_tail40(varargin)
%recompute_baselines_tail40  Recompute all baseline *_tail40.mat files.
%
% This recomputes baselines over the dev/test split (60/40) and saves only
% the tail40 segment as algo-dataset_tail40.mat under the baselines folder.

    script_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(script_dir);

    p = inputParser;
    addParameter(p, 'data_dir', fullfile(repo_root, 'Data Set'));
    addParameter(p, 'out_dir', script_dir);
    addParameter(p, 'olps_dir', 'H:/OLPS-master/Strategy');
    addParameter(p, 'ppt_dir', fullfile(repo_root, '..', 'PPT'));
    addParameter(p, 'tppt_dir', fullfile(repo_root, '..', 'TPPT'));
    addParameter(p, 'tran_cost', 0.001);
    addParameter(p, 'win_size', 5);
    addParameter(p, 'epsilon', 100);
    addParameter(p, 'datasets', {});
    addParameter(p, 'overwrite', true);
    parse(p, varargin{:});
    opts = p.Results;

    if ~exist(opts.out_dir, 'dir')
        mkdir(opts.out_dir);
    end

    addpath(fullfile(repo_root, 'release_ipt_latest', 'code'), '-begin');
    split_cleanup = onCleanup(@() rmpath(fullfile(repo_root, 'release_ipt_latest', 'code')));

    files = dir(fullfile(opts.data_dir, '*.mat'));
    if isempty(files)
        error('No datasets found in %s', opts.data_dir);
    end
    [~, order] = sort({files.name});
    files = files(order);

    target_datasets = lower(string(opts.datasets));

    baselines = {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt'};

    opts_env = struct();
    opts_env.quiet_mode = 1;
    opts_env.display_interval = 1000000;
    opts_env.progress = 0;
    opts_env.log_mode = 0;
    opts_env.mat_mode = 0;
    opts_env.analyze_mode = 0;
    opts_env.his = 0;

    log_path = [tempname, '_olps_log.txt'];
    fid = fopen(log_path, 'w');
    if fid == -1
        error('Cannot open baseline log file: %s', log_path);
    end
    fid_cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    if ~isempty(opts.olps_dir) && exist(opts.olps_dir, 'dir')
        addpath(opts.olps_dir, '-begin');
        olps_cleanup = onCleanup(@() rmpath(opts.olps_dir)); %#ok<NASGU>
    else
        error('OLPS Strategy directory not found: %s', opts.olps_dir);
    end

    for i = 1:numel(files)
        dataset = erase(files(i).name, '.mat');
        if ~isempty(target_datasets) && ~any(lower(string(dataset)) == target_datasets)
            continue;
        end
        data_path = fullfile(opts.data_dir, files(i).name);
        S = load(data_path, 'data');
        data = S.data;
        [T, N] = size(data); %#ok<ASGLU>

        dev = ipt_dev_test_split(T);
        test_start = dev.test_start;
        test_end = dev.test_end;
        data_run = data(1:test_end, :);

        fprintf('\n=== Recompute baselines: %s (T=%d, N=%d), test=%d:%d ===\n', ...
            dataset, T, N, test_start, test_end);

        for bi = 1:numel(baselines)
            algo = baselines{bi};
            out_path = fullfile(opts.out_dir, sprintf('%s-%s_tail40.mat', algo, dataset));
            if ~opts.overwrite && exist(out_path, 'file')
                fprintf('Skip existing: %s\n', out_path);
                continue;
            end

            [daily_ret_full, daily_port_full] = run_baseline_series( ...
                algo, data_run, fid, opts_env, opts);

            daily_ret_full = double(daily_ret_full(:));
            if numel(daily_ret_full) < test_end
                error('baseline=%s returned daily_ret length=%d, expected >=%d', ...
                    algo, numel(daily_ret_full), test_end);
            end

            daily_ret = daily_ret_full(test_start:test_end);
            cumprod_ret = cumprod(daily_ret);
            cum_ret = cumprod_ret(end);

            daily_portfolio = [];
            if ~isempty(daily_port_full)
                daily_portfolio = daily_port_full(test_start:test_end, :);
            end

            if isempty(daily_portfolio)
                save(out_path, 'cum_ret', 'cumprod_ret', 'daily_ret');
            else
                save(out_path, 'cum_ret', 'cumprod_ret', 'daily_ret', 'daily_portfolio');
            end

            fprintf('Saved %s\n', out_path);
        end
    end
end

function [daily_ret_full, daily_port_full] = run_baseline_series(algo, data_run, fid, opts_env, opts)
    tc = opts.tran_cost;
    if strcmpi(algo, 'ubah')
        [~, ~, daily_ret_full, daily_port_full] = ubah_run(fid, data_run, tc, opts_env);
    elseif strcmpi(algo, 'bcrp')
        [~, ~, daily_ret_full, daily_port_full] = bcrp_run(fid, data_run, tc, opts_env);
    elseif strcmpi(algo, 'up')
        [~, ~, daily_ret_full, daily_port_full] = up_run(fid, data_run, tc, opts_env);
    elseif strcmpi(algo, 'olmar2')
        [~, ~, daily_ret_full, daily_port_full] = olmar2_run(fid, data_run, 10, 0.5, tc, opts_env);
    elseif strcmpi(algo, 'rmr')
        [~, ~, daily_ret_full] = rmr_run(fid, data_run, 0.5, tc, 5, opts_env);
        daily_port_full = [];
    elseif strcmpi(algo, 'anticor')
        [~, ~, daily_ret_full, daily_port_full] = anticor_run(fid, data_run, 30, tc, opts_env);
    elseif strcmpi(algo, 'corn')
        [~, ~, daily_ret_full, daily_port_full] = corn_run(fid, data_run, 5, 0.1, tc, opts_env);
    elseif strcmpi(algo, 'ppt')
        [daily_ret_full, daily_port_full] = run_ppt_like_series(opts.ppt_dir, data_run, opts.win_size, opts.epsilon, tc);
    elseif strcmpi(algo, 'tppt')
        [daily_ret_full, daily_port_full] = run_ppt_like_series(opts.tppt_dir, data_run, opts.win_size, opts.epsilon, tc);
    else
        error('Unknown baseline: %s', algo);
    end
end

function [daily_ret_full, port_full] = run_ppt_like_series(model_dir, data, win_size, epsilon, tran_cost)
    if ~exist(model_dir, 'dir')
        error('PPT/TPPT directory not found: %s', model_dir);
    end

    addpath(model_dir, '-begin');
    cleanup = onCleanup(@() rmpath(model_dir)); %#ok<NASGU>
    clear PPT PPT_run simplex_projection_selfnorm2

    [T, N] = size(data);
    close_price = ones(T, N);
    for t = 2:T
        close_price(t, :) = close_price(t - 1, :) .* data(t, :);
    end

    daily_port = ones(N, 1) / N;
    daily_port_o = zeros(N, 1);
    daily_ret_full = ones(T, 1);
    port_full = zeros(T, N);

    for t = 1:T
        port_full(t, :) = daily_port';
        turnover_t = sum(abs(daily_port - daily_port_o));
        daily_ret_full(t, 1) = (data(t, :) * daily_port) * (1 - tran_cost / 2 * turnover_t);

        daily_port_o = daily_port .* data(t, :)' / (data(t, :) * daily_port);

        if t < T
            [daily_port_n, ~, ~] = PPT(close_price, data, t, daily_port, win_size, epsilon);
            daily_port = daily_port_n;
        end
    end
end
