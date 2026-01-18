% Script to run structural ablation on IPT (iptnoturnover version)
% and generate TeX tables for each variant.

% Setup paths
base_dir = fileparts(fileparts(mfilename('fullpath'))); % H:\...\Investment-potential-tracking
addpath(base_dir);
addpath(fullfile(base_dir, 'release_ipt_latest', 'code'));
addpath(fullfile(base_dir, 'baselines')); % Ensure baselines are reachable if needed (though we pass dir)

summary_file = fullfile(base_dir, 'release_ipt_latest', 'results', 'ablation_struct_Qclip10', 'ipt_fixed_log_wealth_Qclip10_summary_dev60_test40_minimal.txt');
results_root = fullfile(base_dir, 'release_ipt_latest', 'results');
baseline_dir = fullfile(base_dir, 'baselines');

% Define variants
variants = struct();

% 1. Base (iptnoturnover reproduction)
variants(1).name = 'ipt_ablation_base';
variants(1).algo_name = 'IPT_Base';
variants(1).force_no_inertia = false;
variants(1).force_no_qclip = false;
variants(1).force_zero_cost = true;

% 2. No Inertia
variants(2).name = 'ipt_ablation_noInertia';
variants(2).algo_name = 'IPT_NoInertia';
variants(2).force_no_inertia = true;
variants(2).force_no_qclip = false;
variants(2).force_zero_cost = true;

% 3. No Q-clip
variants(3).name = 'ipt_ablation_noQclip';
variants(3).algo_name = 'IPT_NoQclip';
variants(3).force_no_inertia = false;
variants(3).force_no_qclip = true;
variants(3).force_zero_cost = true;

% 4. No Inertia & No Q-clip (Raw)
variants(4).name = 'ipt_ablation_raw';
variants(4).algo_name = 'IPT_Raw';
variants(4).force_no_inertia = true;
variants(4).force_no_qclip = true;
variants(4).force_zero_cost = true;

% Standard Baselines for table ordering
baselines = {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt'};

for i = 1:length(variants)
    v = variants(i);
    fprintf('\n========================================\n');
    fprintf('Processing Variant: %s\n', v.name);
    fprintf('========================================\n');
    
    variant_dir = fullfile(results_root, v.name);
    if ~exist(variant_dir, 'dir')
        mkdir(variant_dir);
    end
    
    % 1. Run Export (Generate .mat results)
    fprintf('Generating results...\n');
    run_ablation_export( ...
        'summary_csv', summary_file, ...
        'algo_name', v.algo_name, ...
        'results_dir', variant_dir, ...
        'data_dir', fullfile(base_dir, 'Data Set'), ...
        'force_no_inertia', v.force_no_inertia, ...
        'force_no_qclip', v.force_no_qclip, ...
        'force_zero_cost', v.force_zero_cost ...
    );
    
    % 2. Run Statistical Tests (Generate summary CSVs)
    fprintf('Running Statistical Tests...\n');
    run_statistical_tests( ...
        'results_dir', variant_dir, ...
        'baseline_dir', baseline_dir, ...
        'control_algo', v.algo_name, ...
        'file_pattern', '*_tail40.mat' ...
    );

    % 3. Generate TeX Tables
    fprintf('Generating Tables...\n');
    % Construct algos order: Baselines + Current Variant
    current_algos = [baselines, {v.algo_name}];
    
    % Call ipt_paper_tables
    % Note: we catch output to avoid massive scrolling, but we want the files
    out = ipt_paper_tables( ...
        'results_dir', variant_dir, ...
        'baseline_dir', baseline_dir, ...
        'algos_order', current_algos ...
    );
    
    fprintf('Tables generated in: %s\n', variant_dir);
    
    % Display brief summary
    cw_ranks = mean(out.ranks_cw, 2);
    sr_ranks = mean(out.ranks_sr, 2);
    my_rank_cw = cw_ranks(end);
    my_rank_sr = sr_ranks(end);
    
    fprintf('Variant %s Performance:\n', v.algo_name);
    fprintf('  Mean CW Rank: %.4f\n', my_rank_cw);
    fprintf('  Mean SR Rank: %.4f\n', my_rank_sr);
end

fprintf('\nAll ablations completed.\n');
