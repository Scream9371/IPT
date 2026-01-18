function ipt_ppt_tppt_fixed_test(varargin)
    % ipt_ppt_tppt_fixed_test - Evaluate PPT and TPPT on the Test split with fixed parameters.
    %
    % Fixed parameters (defaults):
    %   win_size = 5
    %   epsilon  = 100
    %   tran_cost = 0.001
    %
    % Split rule (time-based by ratio, default 6/2/2):
    %   train_ratio = 0.6, val_ratio = 0.2, test_ratio = 0.2
    %
    % Outputs:
    %   Investment-potential-tracking/results_fixed_params/ppt_tppt_fixed_test_summary.csv
    %   Investment-potential-tracking/results_fixed_params/ppt_tppt_fixed_test_summary.txt

    p = inputParser;
    addParameter(p, 'win_size', 5);
    addParameter(p, 'epsilon', 100);
    addParameter(p, 'tran_cost', 0.001);
    addParameter(p, 'datasets', []); % [] for all, or e.g. {'ndx', 'tse'} or "ndx"
    addParameter(p, 'train_ratio', 0.6);
    addParameter(p, 'val_ratio', 0.2);
    addParameter(p, 'split_mode', 'train_val_test'); % 'train_val_test' | 'dev_test'
    addParameter(p, 'dev_ratio', 0.6);
    addParameter(p, 'run_tag', ''); % optional suffix to avoid overwriting outputs
    parse(p, varargin{:});
    opts = p.Results;

    script_dir = fileparts(mfilename('fullpath'));
    base_dir = fileparts(script_dir);
    data_dir = fullfile(script_dir, 'Data Set');
    out_dir = fullfile(script_dir, 'results_fixed_params');

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    files = dir(fullfile(data_dir, '*.mat'));

    if isempty(files)
        error('No datasets found in %s', data_dir);
    end

    [~, order] = sort({files.name});
    files = files(order);

    if ~isempty(opts.datasets)

        if ischar(opts.datasets) || isstring(opts.datasets)
            wanted = string(opts.datasets);
        else
            wanted = string(opts.datasets(:));
        end

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

    ppt_dir = fullfile(base_dir, 'PPT');
    tppt_dir = fullfile(base_dir, 'TPPT');

    rows = struct('dataset', {}, 'T', {}, 'N', {}, ...
        'train_end', {}, 'val_start', {}, 'val_end', {}, 'test_start', {}, 'test_end', {}, ...
        'train_ratio', {}, 'val_ratio', {}, 'test_ratio', {}, ...
        'split_mode', {}, 'dev_ratio', {}, ...
        'tran_cost', {}, 'win_size', {}, 'epsilon', {}, 'ppt_test_wealth', {}, 'tppt_test_wealth', {});

    for i = 1:numel(files)
        dataset = erase(files(i).name, '.mat');
        data_path = fullfile(data_dir, files(i).name);
        S = load(data_path, 'data');
        data = S.data;
        [T, N] = size(data);

        split_mode = lower(string(opts.split_mode));

        if split_mode ~= "train_val_test" && split_mode ~= "dev_test"
            error('Unsupported split_mode: %s (use train_val_test or dev_test)', split_mode);
        end

        if split_mode == "train_val_test"
            split = ipt_time_split_ends(T, 'train_ratio', opts.train_ratio, 'val_ratio', opts.val_ratio);
        else
            dev = ipt_dev_test_split(T, 'dev_ratio', opts.dev_ratio);
            split = struct();
            split.train_end = dev.dev_end; % warm-up only
            split.val_start = dev.test_start; % not used
            split.val_end = dev.test_start - 1; % empty
            split.test_start = dev.test_start;
            split.test_end = dev.test_end;
            split.train_ratio = dev.dev_ratio;
            split.val_ratio = 0;
            split.test_ratio = dev.test_ratio;
        end

        fprintf('\n=== Fixed-param test: %s (T=%d, N=%d) ===\n', dataset, T, N);
        fprintf('Params: win_size=%d, epsilon=%.1f, tran_cost=%.6f\n', opts.win_size, opts.epsilon, opts.tran_cost);

        if split_mode == "dev_test"
            fprintf('Split(dev/test): dev=1:%d, test=%d:%d (ratios %.2f/%.2f)\n', ...
                split.train_end, split.test_start, split.test_end, split.train_ratio, split.test_ratio);
        else
            fprintf('Split: train=1:%d, val=%d:%d, test=%d:%d (ratios %.2f/%.2f/%.2f)\n', ...
                split.train_end, split.val_start, split.val_end, split.test_start, split.test_end, ...
                split.train_ratio, split.val_ratio, split.test_ratio);
        end

        ppt_test_wealth = eval_with_model_dir(ppt_dir, data, opts.win_size, opts.epsilon, opts.tran_cost, split.test_start, split.test_end);
        fprintf('PPT  test_wealth=%.10f\n', ppt_test_wealth);

        tppt_test_wealth = eval_with_model_dir(tppt_dir, data, opts.win_size, opts.epsilon, opts.tran_cost, split.test_start, split.test_end);
        fprintf('TPPT test_wealth=%.10f\n', tppt_test_wealth);

        entry = struct();
        entry.dataset = dataset;
        entry.T = T;
        entry.N = N;
        entry.train_end = split.train_end;
        entry.val_start = split.val_start;
        entry.val_end = split.val_end;
        entry.test_start = split.test_start;
        entry.test_end = split.test_end;
        entry.train_ratio = split.train_ratio;
        entry.val_ratio = split.val_ratio;
        entry.test_ratio = split.test_ratio;
        entry.split_mode = char(split_mode);
        entry.dev_ratio = opts.dev_ratio;
        entry.tran_cost = opts.tran_cost;
        entry.win_size = opts.win_size;
        entry.epsilon = opts.epsilon;
        entry.ppt_test_wealth = ppt_test_wealth;
        entry.tppt_test_wealth = tppt_test_wealth;
        rows(end + 1) = entry;
    end

    Tsum = struct2table(rows);

    if lower(string(opts.split_mode)) == "dev_test"
        tag = sprintf('dev%.0f_test%.0f', 100 * opts.dev_ratio, 100 * (1 - opts.dev_ratio));
    else
        tag = sprintf('%.0f_%.0f_%.0f', 100 * opts.train_ratio, 100 * opts.val_ratio, 100 * (1 - opts.train_ratio - opts.val_ratio));
    end

    run_tag = string(opts.run_tag);

    if strlength(run_tag) > 0
        run_tag = "_" + run_tag;
    end

    csv_path = fullfile(out_dir, "ppt_tppt_fixed_test_summary_" + tag + run_tag + ".csv");
    writetable(Tsum, csv_path);

    txt_path = fullfile(out_dir, "ppt_tppt_fixed_test_summary_" + tag + run_tag + ".txt");
    fid = fopen(txt_path, 'w');

    if fid ~= -1
        fprintf(fid, '%s\n', strjoin(Tsum.Properties.VariableNames, '\t'));

        for i = 1:height(Tsum)
            row = Tsum(i, :);
            parts = cell(1, width(Tsum));

            for j = 1:width(Tsum)
                v = row{1, j};

                if isnumeric(v)
                    parts{j} = num2str(v, '%.10g');
                else
                    parts{j} = string(v);
                end

            end

            fprintf(fid, '%s\n', strjoin(string(parts), '\t'));
        end

        fclose(fid);
    end

    fprintf('\nSaved: %s\nSaved: %s\n', csv_path, txt_path);
end

function wealth = eval_with_model_dir(model_dir, data, win_size, epsilon, tran_cost, start_idx, end_idx)
    addpath(model_dir, '-begin');
    clear PPT PPT_run simplex_projection_selfnorm2

    [T, N] = size(data);

    if end_idx > T
        error('end_idx out of range');
    end

    close_price = ones(T, N);

    for i = 2:T
        close_price(i, :) = close_price(i - 1, :) .* data(i, :);
    end

    daily_port = ones(N, 1) / N;
    daily_port_o = zeros(N, 1);
    wealth = 1;

    for t = 1:end_idx
        daily_incre = (data(t, :) * daily_port) * (1 - tran_cost / 2 * sum(abs(daily_port - daily_port_o)));

        if t >= start_idx
            wealth = wealth * daily_incre;
        end

        daily_port_o = daily_port .* data(t, :)' / (data(t, :) * daily_port);

        if t < end_idx
            [daily_port_n, ~, ~] = PPT(close_price, data, t, daily_port, win_size, epsilon);
            daily_port = daily_port_n;
        end

    end

    rmpath(model_dir);
    clear PPT PPT_run simplex_projection_selfnorm2
end
