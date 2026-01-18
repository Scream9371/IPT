function run_ablation_gate_study()
% Run ablation study for gate/cap mechanism
% Compare 4 configurations:
% A0: No gate, no cap (baseline)
% A1: Only action gate (defensive_active controls Q in action)
% A2: Only selective cap (cap_active)
% A3: Gate + cap (full mechanism)
%
% Datasets: NDX, NYSE-O (trending), SZ50, INV500 (downside/weak)

    % Define datasets
    datasets = {'ndx', 'nyse-o', 'sz50', 'inv500'};

    % Define ablation configurations
    configs = struct(...
        'name', {'A0_no_gate_no_cap', 'A1_action_gate', 'A2_selective_cap', 'A3_full'}, ...
        'risk_cap_on_gate', {false, false, true, true}, ...
        'defensive_active', {false, true, false, true} ...
    );

    % Base parameters
    base_params = struct(...
        'win_size', 5, ...
        'epsilon', 10, ...
        'update_mix', 0.5, ...
        'max_turnover', 0.50, ...
        'tran_cost', 0.001, ...
        'val_objective', 'wins_both', ...
        'use_parallel', true, ...
        'num_workers', 4, ...
        'sharpe_annualization', 252 ...
    );

    % Results storage
    results = struct();

    fprintf('Starting ablation study for gate/cap mechanism...\n');
    fprintf('Datasets: %s\n', strjoin(datasets, ', '));
    fprintf('Configurations: %s\n', strjoin({configs.name}, ', '));
    fprintf('\n');

    for d = 1:length(datasets)
        dataset = datasets{d};
        fprintf('Processing dataset: %s\n', dataset);

        for c = 1:length(configs)
            config = configs(c);
            fprintf('  Config: %s\n', config.name);

            % Set parameters
            params = base_params;
            params.risk_cap_on_gate = config.risk_cap_on_gate;

            % Run IPT
            try
                % Load data
                data_file = fullfile('Data Set', [dataset, '.mat']);
                if exist(data_file, 'file')
                    data_struct = load(data_file);
                    if isfield(data_struct, 'data')
                        data = data_struct.data;
                    else
                        fields = fieldnames(data_struct);
                        data = data_struct.(fields{1});
                    end
                else
                    error('Data file not found: %s', data_file);
                end

                % Run evaluation using ipt_fixed_test
                [daily_port, daily_ret, cum_ret, wealth, sharpe, max_drawdown, turnover_mean, ~, summary] = ipt_fixed_test(...
                    data, params.win_size, params.epsilon, params.update_mix, ...
                    params.max_turnover, params.tran_cost, params.sharpe_annualization, ...
                    false, 0, false, params.risk_cap_on_gate, struct(), ...
                    1, size(data,1), params.use_parallel, params.num_workers);

                % Store results
                result_key = sprintf('%s_%s', dataset, config.name);
                results.(result_key) = struct(...
                    'dataset', dataset, ...
                    'config', config.name, ...
                    'wealth', wealth, ...
                    'max_drawdown', max_drawdown, ...
                    'turnover_mean', turnover_mean, ...
                    'sharpe', sharpe, ...
                    'success', true ...
                );

                fprintf('    Success: CW=%.4f, Sharpe=%.4f\n', wealth, sharpe);

            catch ME
                fprintf('    Error: %s\n', ME.message);
                results.(sprintf('%s_%s', dataset, config.name)) = struct(...
                    'dataset', dataset, ...
                    'config', config.name, ...
                    'success', false, ...
                    'error', ME.message ...
                );
            end
        end
        fprintf('\n');
    end

    % Save results
    output_file = fullfile('results_fixed_params', 'ablation_gate_study_results.mat');
    save(output_file, 'results', 'datasets', 'configs');
    fprintf('Results saved to: %s\n', output_file);

    % Generate summary report
    generate_ablation_report(results, datasets, configs);
end

function generate_ablation_report(results, datasets, configs)
    fprintf('\n=== Ablation Study Summary ===\n');

    % Create summary table
    summary = table();
    for d = 1:length(datasets)
        dataset = datasets{d};
        for c = 1:length(configs)
            config = configs(c);
            key = sprintf('%s_%s', dataset, config.name);
            if isfield(results, key) && results.(key).success
                summary = [summary; {
                    dataset, ...
                    config.name, ...
                    results.(key).wealth, ...
                    results.(key).sharpe, ...
                    results.(key).turnover_mean, ...
                    results.(key).max_drawdown ...
                }];
            end
        end
    end

    if ~isempty(summary)
        summary.Properties.VariableNames = {'Dataset', 'Config', 'Wealth', 'Sharpe', 'Turnover', 'MaxDD'};
        writetable(summary, fullfile('results_fixed_params', 'ablation_summary.csv'));
        fprintf('Summary saved to: results_fixed_params/ablation_summary.csv\n');

        % Display summary
        fprintf('\nPerformance Summary:\n');
        disp(summary);
    end
end
