function out = ipt_analyze_results(varargin)
    %IPT_ANALYZE_RESULTS  Analyze run_ipt results (stats + tables).
    %
    % This helper:
    %   1. Ensures the results_dir contains baselines (symlink or copy if needed,
    %      but run_statistical_tests handles separate baseline_dir).
    %   2. Calls run_statistical_tests to generate stat_*.csv.
    %   3. Calls ipt_paper_tables to generate TeX tables.
    %
    % Usage:
    %   ipt_analyze_results('results_dir', '...', 'baseline_dir', '...', 'algos_order', {...});

    p = inputParser;
    addParameter(p, 'results_dir', '');
    addParameter(p, 'baseline_dir', '');
    addParameter(p, 'datasets_order', {'djia', 'inv500', 'marpd', 'msci', 'nyse-n', 'nyse-o', 'tse'});
    addParameter(p, 'algos_order', {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt'}); % Will append current run algo if not present
    addParameter(p, 'control_algo', 'ipt'); % For statistical tests (CD plot)
    addParameter(p, 'alpha', 0.05);
    parse(p, varargin{:});
    opts = p.Results;

    results_dir = char(string(opts.results_dir));
    baseline_dir = char(string(opts.baseline_dir));

    if isempty(results_dir) || ~exist(results_dir, 'dir')
        error('results_dir must be provided and exist.');
    end

    if isempty(baseline_dir) || ~exist(baseline_dir, 'dir')
        % Try to infer baseline_dir if not provided
        % Assuming standard repo structure: .../release_ipt_latest/code/../../baselines
        % run_ipt usually runs in release_ipt_latest/code
        possible_base = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'baselines');

        if exist(possible_base, 'dir')
            baseline_dir = possible_base;
        else
            warning('baseline_dir not provided and could not be inferred. Statistical tests may fail if baselines are missing.');
        end

    end

    % 1. Identify the IPT algo name from results_dir
    % run_ipt generates mat files like: ipt_<struct_name>-<dataset>.mat
    % We need to find the struct name.
    files = dir(fullfile(results_dir, 'ipt_*.mat'));
    ipt_algo_name = '';

    if ~isempty(files)
        % parse first file: ipt_NAME-DATASET.mat
        % or ipt_NAME-DATASET-TAG.mat
        % Simplest way: extract everything between 'ipt_' and '-dataset'
        % But dataset names can contain hyphens (nyse-n).
        % Let's rely on the structure names used in run_ipt.
        % Actually, run_ipt outputs per-structure folders? No, looking at run_ipt code:
        % sdir = fullfile(results_root, s.name);
        % save(fullfile(sdir, out_name), ...)
        % So run_ipt puts mat files in SUBDIRECTORIES named by structure.
        % AND it puts run_ipt_results_*.mat in the run_dir root.

        % Wait, ipt_analyze_results is likely called on 'run_dir' (e.g. run_v1_base_...).
        % Inside run_dir, there are subfolders for each structure?
        % Let's check run_ipt.m line 104: sdir = fullfile(run_dir, s.name);
        % So yes, mat files are in subfolders.

        % However, run_statistical_tests expects all mat files in one folder (results_dir).
        % So we need to flatten or point to the specific structure folder.

        % If run_ipt was run with multiple structures, we might have multiple subfolders.
        % For now, let's assume we want to analyze ALL structures found in run_dir.
        % But run_statistical_tests takes one folder.
        % If we have multiple structures, we might want to compare them all + baselines.
        % Or maybe this function should be called per-structure?

        % Strategy:
        % 1. Find all subdirectories in results_dir that contain ipt_*.mat
        % 2. For each such folder, it effectively represents an algorithm variant.
        % 3. We can either:
        %    a) Run stats for each variant folder separately vs baselines.
        %    b) Copy/Symlink all variant mat files into a single 'merged' folder to compare variants vs each other vs baselines.

        % Given the user wants "Sharpe/Calmar... for 7 datasets", likely for the run they just executed.
        % Usually run_ipt is run with one structure (like v1_base).
        % So there should be one subfolder matching structure name.

        subdirs = dir(results_dir);
        subdirs = subdirs([subdirs.isdir]);
        subdirs = subdirs(~ismember({subdirs.name}, {'.', '..'}));

        valid_dirs = {};
        algo_names = {};

        for i = 1:numel(subdirs)
            dname = subdirs(i).name;
            dpath = fullfile(results_dir, dname);

            if ~isempty(dir(fullfile(dpath, 'ipt_*.mat')))
                valid_dirs{end + 1} = dpath;
                algo_names{end + 1} = dname;
            end

        end

        if isempty(valid_dirs)
            % Check if mat files are in root (older behavior or simple run)
            if ~isempty(dir(fullfile(results_dir, 'ipt_*.mat')))
                valid_dirs = {results_dir};
                % Try to guess algo name from filename
                f = dir(fullfile(results_dir, 'ipt_*.mat'));
                % ipt_NAME-DATASET...
                tokens = regexp(f(1).name, 'ipt_([^-]+)-', 'tokens');

                if ~isempty(tokens)
                    algo_names = {tokens{1}{1}};
                else
                    algo_names = {'ipt'};
                end

            else
                error('No ipt_*.mat files found in %s or its subdirectories.', results_dir);
            end

        end

        % We will run analysis for the FIRST valid directory found (primary use case: single variant run).
        % If multiple, we might warn.
        if numel(valid_dirs) > 1
            warning('Multiple structure folders found. Analyzing the first one: %s', algo_names{1});
        end

        target_dir = valid_dirs{1};
        target_algo = algo_names{1};

        % We need to make sure the mat files have standard naming for run_statistical_tests?
        % run_statistical_tests expects: <algo>-<dataset>_tail40.mat (OLPS style)
        % OR run_ipt outputs: ipt_<struct>-<dataset>[-tag].mat
        % run_statistical_tests:
        %   [algo, dataset] = parse_algo_dataset(name);
        %   parse_algo_dataset usually splits by '-'.
        %
        %   ipt_v1_base-djia-v1_base.mat -> algo="ipt_v1_base", dataset="djia" (if logic holds)
        %
        % Let's verify run_statistical_tests parsing logic (it's not visible here but assuming standard split).
        % If run_ipt outputs 'ipt_v1_base-djia.mat', then algo='ipt_v1_base', dataset='djia'.
        % This should work.

        % 2. Run Statistical Tests
        fprintf('Running Statistical Tests on %s...\n', target_dir);

        % Construct algos list including baselines and our target
        full_algos = opts.algos_order;

        if ~ismember(target_algo, full_algos)
            full_algos = [full_algos, {target_algo}];
        end

        % Ensure control_algo is present (default 'ipt' might not match target_algo)
        ctrl = opts.control_algo;

        if strcmpi(ctrl, 'ipt') && ~ismember('ipt', full_algos)
            % If user said control='ipt' but our algo is 'v1_base', use 'v1_base'
            ctrl = target_algo;
        end

        run_statistical_tests( ...
            'results_dir', target_dir, ...
            'baseline_dir', baseline_dir, ...
            'alpha', opts.alpha, ...
            'control_algo', ctrl, ...
            'exclude_datasets', {}, ... % run_ipt usually filters datasets already
            'file_pattern', '*.mat' ... % Match all mat files
        );

        % 3. Generate Tables
        fprintf('Generating Tables for %s...\n', target_dir);

        % Need to map algo names to display names if possible
        % For now, just use raw names or simple capitalization
        algo_disp = full_algos;

        for k = 1:numel(algo_disp)

            if strcmpi(algo_disp{k}, 'ipt') || strcmpi(algo_disp{k}, target_algo)
                algo_disp{k} = 'IPT';
            else
                algo_disp{k} = upper(algo_disp{k});
            end

        end

        out = ipt_paper_tables( ...
            'results_dir', target_dir, ...
            'baseline_dir', baseline_dir, ...
            'datasets_order', opts.datasets_order, ...
            'algos_order', full_algos, ...
            'algo_display', algo_disp, ...
            'alpha', opts.alpha ...
        );

        % Display brief summary (Rankings)
        if isfield(out, 'ranks_cw') && isfield(out, 'ranks_sr')
            cw_ranks = mean(out.ranks_cw, 2);
            sr_ranks = mean(out.ranks_sr, 2);
            % Assuming the last algo is ours (if appended)
            % Or find index of target_algo
            [~, idx] = ismember(target_algo, full_algos);

            if idx > 0
                my_rank_cw = cw_ranks(idx);
                my_rank_sr = sr_ranks(idx);

                fprintf('\n=====================================\n');
                fprintf('Performance Summary for %s\n', target_algo);
                fprintf('=====================================\n');
                fprintf('  Mean CW Rank: %.4f\n', my_rank_cw);
                fprintf('  Mean SR Rank: %.4f\n', my_rank_sr);
                fprintf('=====================================\n');
            end

        end

        % 4. Save Aggregated CSV
        % Merge metrics into a single table:
        % [Dataset, Algo, CW, APY, Sharpe, Calmar, CW_Rank, SR_Rank]
        % (Simplified: Just save what we have from out struct)

        fprintf('Saving aggregated metrics CSV...\n');

        % We have: out.cw, out.apy, out.sr, out.calmar (n_algos x n_datasets)
        % out.ranks_cw, out.ranks_sr (n_algos x n_datasets)
        % datasets = opts.datasets_order (columns)
        % algos = full_algos (rows)

        datasets = opts.datasets_order;
        algos = full_algos;
        n_algos = numel(algos);
        n_datasets = numel(datasets);

        T_agg = table();

        for i = 1:n_algos
            a = algos{i};

            for j = 1:n_datasets
                d = datasets{j};

                % Check if index is valid (out struct might only contain datasets found)
                % But ipt_paper_tables usually handles the full grid defined by datasets_order.

                row = table();
                row.Algo = string(a);
                row.Dataset = string(d);
                row.CW = out.cw(i, j);
                row.APY = out.apy(i, j);
                row.Sharpe = out.sr(i, j);
                row.Calmar = out.calmar(i, j);
                row.Rank_CW = out.ranks_cw(i, j);
                row.Rank_SR = out.ranks_sr(i, j);

                T_agg = [T_agg; row];
            end

        end

        csv_name = sprintf('metrics_summary_%s.csv', target_algo);
        writetable(T_agg, fullfile(target_dir, csv_name));
        fprintf('Saved metrics CSV: %s\n', fullfile(target_dir, csv_name));

        fprintf('Analysis completed. Tables saved in %s\n', target_dir);

    else
        warning('No ipt results found to analyze.');
        out = [];
    end

end
