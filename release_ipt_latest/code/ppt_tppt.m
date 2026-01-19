function ppt_tppt(varargin)
    % ppt_tppt - Evaluate PPT and TPPT on the Test split with fixed parameters.
    %
    % Fixed parameters:
    %   win_size = 5
    %   tran_cost = 0.001
    %
    % Split rule:
    %   dev/test = 0.6/0.4, and the model is run ONLY on the test segment
    %   (i.e., it starts fresh at test_start with uniform weights).
    %
    % Outputs:
    %   Investment-potential-tracking/results_fixed_params/ppt_tppt_fixed_test_summary.csv
    %   Investment-potential-tracking/results_fixed_params/ppt_tppt_fixed_test_summary.txt

    p = inputParser;
    addParameter(p, 'win_size', 5);
    addParameter(p, 'tran_cost', 0.001);
    addParameter(p, 'datasets', []); % [] for all, or e.g. {'ndx', 'tse'} or "ndx"
    addParameter(p, 'run_tag', ''); % optional suffix to avoid overwriting outputs
    parse(p, varargin{:});
    opts = p.Results;

    script_dir = fileparts(mfilename('fullpath'));
    data_dir = fullfile(script_dir, '..', '..', 'Data Set');
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

    ppt_dir = local_find_model_dir('PPT', script_dir);
    tppt_dir = local_find_model_dir('TPPT', script_dir);

    rows = struct('dataset', {}, 'T', {}, 'N', {}, ...
        'test_start', {}, 'test_end', {}, ...
        'tran_cost', {}, 'win_size', {}, 'ppt_test_wealth', {}, 'tppt_test_wealth', {});

    for i = 1:numel(files)
        dataset = erase(files(i).name, '.mat');
        data_path = fullfile(data_dir, files(i).name);
        S = load(data_path, 'data');
        data = S.data;
        [T, N] = size(data);
        dev_end = floor(T * 0.6);
        test_start = dev_end + 1;
        test_end = T;
        data_test = data(test_start:test_end, :);

        fprintf('\n=== Fixed-param test: %s (T=%d, N=%d) ===\n', dataset, T, N);
        fprintf('Params: win_size=%d, tran_cost=%.6f\n', opts.win_size, opts.tran_cost);
        fprintf('Split(dev/test): dev=1:%d, test=%d:%d\n', dev_end, test_start, test_end);

        ppt_test_wealth = eval_ppt_like_on_test_only(ppt_dir, data_test, opts.win_size, opts.tran_cost);
        fprintf('PPT  test_wealth=%.10f\n', ppt_test_wealth);

        tppt_test_wealth = eval_ppt_like_on_test_only(tppt_dir, data_test, opts.win_size, opts.tran_cost);
        fprintf('TPPT test_wealth=%.10f\n', tppt_test_wealth);

        entry = struct();
        entry.dataset = dataset;
        entry.T = T;
        entry.N = N;
        entry.test_start = test_start;
        entry.test_end = test_end;
        entry.tran_cost = opts.tran_cost;
        entry.win_size = opts.win_size;
        entry.ppt_test_wealth = ppt_test_wealth;
        entry.tppt_test_wealth = tppt_test_wealth;
        rows(end + 1) = entry;
    end

    Tsum = struct2table(rows);

    tag = 'dev60_test40';

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

function wealth = eval_ppt_like_on_test_only(model_dir, data_test, win_size, tran_cost)
    addpath(model_dir, '-begin');
    clear PPT PPT_run simplex_projection_selfnorm2
    [cum_wealth, ~, ~] = PPT_run(data_test, win_size, tran_cost);
    wealth = cum_wealth(end);

    rmpath(model_dir);
    clear PPT PPT_run simplex_projection_selfnorm2
end

function model_dir = local_find_model_dir(model_kind, script_dir)
    model_dir = fullfile(script_dir, '..', '..', '..', model_kind);

    if ~exist(model_dir, 'dir')
        model_dir = fullfile(script_dir, '..', '..', model_kind);
    end

    if ~exist(model_dir, 'dir')
        error('Model dir not found for %s (searched: %s)', model_kind, model_dir);
    end

end
