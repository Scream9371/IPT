% run_micro_experiments.m
% 执行 IPT 的 3 条微改进实验，并生成对应的 LaTeX 表格片段。
%
% 使用方式：
%   1）在 MATLAB 中将当前工作目录切换到 Investment-potential-tracking 根目录：
%          cd('h:/20250808_research/IPT/matlab code/Investment-potential-tracking')
%      然后运行：
%          run('scripts/run_micro_experiments.m')
%
%   2）或在命令行中使用 batch 模式：
%          matlab -batch "cd('h:/20250808_research/IPT/matlab code/Investment-potential-tracking'); run('scripts/run_micro_experiments.m');"
%
% 运行完成后：
%   - 三条实验结果分别保存在：
%       release_ipt_latest/results/ipt_micro_objective
%       release_ipt_latest/results/ipt_micro_inertia
%       release_ipt_latest/results/ipt_micro_qsmooth
%   - 对应的 LaTeX 表格片段保存在：
%       docs/paper_tables/paper_tables_micro_objective.tex
%       docs/paper_tables/paper_tables_micro_inertia.tex
%       docs/paper_tables/paper_tables_micro_qsmooth.tex
%
% Add paths
script_path = mfilename('fullpath');
root_dir = fileparts(fileparts(script_path)); % Investment-potential-tracking root
code_dir = fullfile(root_dir, 'release_ipt_latest', 'code');
tools_dir = fullfile(root_dir, 'release_ipt_latest', 'tools');
results_root = fullfile(root_dir, 'release_ipt_latest', 'results');
matlab_code_dir = fileparts(root_dir); % Parent of Investment-potential-tracking

addpath(code_dir);
addpath(tools_dir);
addpath(root_dir); % for ipt_paper_tables
addpath(fullfile(matlab_code_dir, 'PPT'));
addpath(fullfile(matlab_code_dir, 'TPPT'));

datasets_list = {'djia', 'inv500', 'msci', 'ndx', 'nyse-n', 'nyse-o', 'sz50', 'tse'};
data_dir = fullfile(root_dir, 'Data Set');
baseline_dir = fullfile(root_dir, 'baselines');
paper_tables_dir = fullfile(root_dir, 'docs', 'paper_tables');

% Experiment 1: Objective Change (Stable)
fprintf('\n=======================================================\n');
fprintf('Running Experiment 1: Objective Change (log_wealth_stable)...\n');
out_dir_1 = 'ipt_micro_objective';
abs_out_dir_1 = fullfile(results_root, out_dir_1);
ipt_fixed_test('val_objective', 'log_wealth_stable', 'out_dir_name', abs_out_dir_1, 'grid_profile', 'minimal', 'datasets', datasets_list, 'data_dir', data_dir);

results_dir_1 = abs_out_dir_1;
csv_path_1 = find_summary_csv(results_dir_1, 'log_wealth_stable');

fprintf('Copying baselines from %s to %s...\n', baseline_dir, results_dir_1);
copy_tail40_baselines(baseline_dir, results_dir_1, datasets_list);

fprintf('Exporting IPT results to %s...\n', results_dir_1);
export_fixed_ipt_results('summary_csv', csv_path_1, 'results_dir', results_dir_1, 'algo_name', 'IPT', 'adaptive_inertia_q', false, 'Q_smoothing_alpha', 0, 'data_dir', data_dir);

fprintf('Running statistical tests for Experiment 1...\n');
run_statistical_tests('results_dir', results_dir_1, 'control_algo', 'ipt');

fprintf('Generating tables for Experiment 1...\n');
ipt_paper_tables('results_dir', results_dir_1, ...
    'algos_order', {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt', 'ipt'}, ...
    'algo_display', {'UBAH', 'BCRP', 'UP', 'OLMAR-2', 'RMR', 'Anticor', 'CORN', 'PPT', 'TPPT', 'IPT'});
snippet_1 = fullfile(results_dir_1, 'paper_tables_plus_djia_snippet.tex');

if exist(snippet_1, 'file')

    if ~exist(paper_tables_dir, 'dir')
        mkdir(paper_tables_dir);
    end

    copyfile(snippet_1, fullfile(paper_tables_dir, 'paper_tables_micro_objective.tex'));
end

% Experiment 2: Adaptive Inertia
fprintf('\n=======================================================\n');
fprintf('Running Experiment 2: Adaptive Inertia (adaptive_inertia_q=true)...\n');
out_dir_2 = 'ipt_micro_inertia';
abs_out_dir_2 = fullfile(results_root, out_dir_2);
% Note: Using default objective 'wins_both' but enabling inertia
ipt_fixed_test('val_objective', 'wins_both', 'adaptive_inertia_q', true, 'out_dir_name', abs_out_dir_2, 'grid_profile', 'minimal', 'datasets', datasets_list, 'data_dir', data_dir);

results_dir_2 = abs_out_dir_2;
csv_path_2 = find_summary_csv(results_dir_2, 'wins_both');

fprintf('Copying baselines from %s to %s...\n', baseline_dir, results_dir_2);
copy_tail40_baselines(baseline_dir, results_dir_2, datasets_list);

fprintf('Exporting IPT results to %s...\n', results_dir_2);
export_fixed_ipt_results('summary_csv', csv_path_2, 'results_dir', results_dir_2, 'algo_name', 'IPT', 'adaptive_inertia_q', true, 'Q_smoothing_alpha', 0, 'data_dir', data_dir);

fprintf('Running statistical tests for Experiment 2...\n');
run_statistical_tests('results_dir', results_dir_2, 'control_algo', 'ipt');

fprintf('Generating tables for Experiment 2...\n');
ipt_paper_tables('results_dir', results_dir_2, ...
    'algos_order', {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt', 'ipt'}, ...
    'algo_display', {'UBAH', 'BCRP', 'UP', 'OLMAR-2', 'RMR', 'Anticor', 'CORN', 'PPT', 'TPPT', 'IPT'});
snippet_2 = fullfile(results_dir_2, 'paper_tables_plus_djia_snippet.tex');

if exist(snippet_2, 'file')

    if ~exist(paper_tables_dir, 'dir')
        mkdir(paper_tables_dir);
    end

    copyfile(snippet_2, fullfile(paper_tables_dir, 'paper_tables_micro_inertia.tex'));
end

% Experiment 3: Q Smoothing
fprintf('\n=======================================================\n');
fprintf('Running Experiment 3: Q Smoothing (Q_smoothing_alpha=0.2)...\n');
out_dir_3 = 'ipt_micro_qsmooth';
abs_out_dir_3 = fullfile(results_root, out_dir_3);
% Note: Using default objective 'wins_both' but enabling Q smoothing
ipt_fixed_test('val_objective', 'wins_both', 'Q_smoothing_alpha', 0.2, 'out_dir_name', abs_out_dir_3, 'grid_profile', 'minimal', 'datasets', datasets_list, 'data_dir', data_dir);

results_dir_3 = abs_out_dir_3;
csv_path_3 = find_summary_csv(results_dir_3, 'wins_both');

fprintf('Copying baselines from %s to %s...\n', baseline_dir, results_dir_3);
copy_tail40_baselines(baseline_dir, results_dir_3, datasets_list);

fprintf('Exporting IPT results to %s...\n', results_dir_3);
export_fixed_ipt_results('summary_csv', csv_path_3, 'results_dir', results_dir_3, 'algo_name', 'IPT', 'adaptive_inertia_q', false, 'Q_smoothing_alpha', 0.2, 'data_dir', data_dir);

fprintf('Running statistical tests for Experiment 3...\n');
run_statistical_tests('results_dir', results_dir_3, 'control_algo', 'ipt');

fprintf('Generating tables for Experiment 3...\n');
ipt_paper_tables('results_dir', results_dir_3, ...
    'algos_order', {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt', 'ipt'}, ...
    'algo_display', {'UBAH', 'BCRP', 'UP', 'OLMAR-2', 'RMR', 'Anticor', 'CORN', 'PPT', 'TPPT', 'IPT'});
snippet_3 = fullfile(results_dir_3, 'paper_tables_plus_djia_snippet.tex');

if exist(snippet_3, 'file')

    if ~exist(paper_tables_dir, 'dir')
        mkdir(paper_tables_dir);
    end

    copyfile(snippet_3, fullfile(paper_tables_dir, 'paper_tables_micro_qsmooth.tex'));
end

fprintf('\nAll experiments completed successfully.\n');

function copy_tail40_baselines(baseline_dir, target_dir, datasets_list)
    algos = {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt'};

    if ~exist(target_dir, 'dir')
        mkdir(target_dir);
    end

    for i = 1:numel(datasets_list)
        ds = datasets_list{i};

        for j = 1:numel(algos)
            algo = algos{j};
            src = fullfile(baseline_dir, sprintf('%s-%s_tail40.mat', algo, ds));
            dst = fullfile(target_dir, sprintf('%s-%s_tail40.mat', algo, ds));

            if exist(src, 'file')
                copyfile(src, dst);
            else
                warning('Baseline file not found: %s', src);
            end

        end

    end

end

function csv_path = find_summary_csv(results_dir, objective)
    % Helper to find the summary CSV file in the results directory.
    % ipt_fixed_test appends params to the filename, so we search by pattern.

    pattern = fullfile(results_dir, sprintf('ipt_fixed_*%s*_summary_*.csv', objective));
    files = dir(pattern);

    if isempty(files)
        % Try simpler pattern
        pattern = fullfile(results_dir, sprintf('ipt_fixed_%s_summary.csv', objective));
        files = dir(pattern);
    end

    if isempty(files)
        % Try listing all csvs
        pattern = fullfile(results_dir, '*.csv');
        files = dir(pattern);
    end

    if isempty(files)
        error('No summary CSV found in %s for objective %s', results_dir, objective);
    end

    % Pick the most recent one if multiple
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
    fprintf('Found summary CSV: %s\n', files(idx).name);
end
