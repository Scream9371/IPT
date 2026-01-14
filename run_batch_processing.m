function run_batch_processing()
    % Batch processing script for IPT algorithm on all datasets
    % This script runs the IPT model on all available datasets and saves results
    % Generates a summary CSV file with final cumulative wealth for each dataset

    % Initialize data structure for CSV output
    csv_data = struct();

    % Define CSV header with 15 fields as specified
    csv_header = {'inv30-A', 'inv30-B', 'inv30-C', 'inv500-A', 'inv500-B', 'inv500-C', ...
                      'NDX-A', 'NDX-B', 'NDX-C', 'weight_inspect_wins', 'risk_inspect_wins', ...
                      'tran_cost', 'reverse_factor', 'risk_factor', 'q_value'};

    datasets = {
                'Data Set/djia.mat',
                'Data Set/inv30-A.mat',
                'Data Set/inv30-B.mat',
                'Data Set/inv30-C.mat',
                'Data Set/inv500-A.mat',
                'Data Set/inv500-B.mat',
                'Data Set/inv500-C.mat',
                'Data Set/msci.mat',
                'Data Set/NDX-A.mat',
                'Data Set/NDX-B.mat',
                'Data Set/NDX-C.mat',
                'Data Set/nyse-n.mat',
                'Data Set/nyse-o.mat',
                'Data Set/tse.mat'
                };

    for i = 1:length(datasets)
        dataset_path = datasets{i};
        dataset_name = strrep(dataset_path, 'Data Set/', '');
        dataset_name = strrep(dataset_name, '.mat', '');

        fprintf('Processing dataset: %s\n', dataset_name);

        % % Modify parameters based on dataset characteristics
        % [~, name, ~] = fileparts(dataset_path);
        % if contains(name, 'inv30')
        %     weight_inspect_wins = 45;
        %     risk_inspect_wins = 15;
        % elseif contains(name, 'inv500')
        %     weight_inspect_wins = 60;
        %     risk_inspect_wins = 20;
        % elseif contains(name, 'NDX')
        %     weight_inspect_wins = 60;
        %     risk_inspect_wins = 20;
        % else
        %     weight_inspect_wins = 252;
        %     risk_inspect_wins = 84;
        % end
        weight_inspect_wins = 63;
        risk_inspect_wins = 21;
        tran_cost = 0.00;
        reverse_factor = 10;
        risk_factor = 50;
        q_value = 0.4;
        L_percentile = 95;

        % Run the IPT algorithm
        try
            [cum_wealth, daily_incre_fact, b_history] = run_ipt(dataset_path, weight_inspect_wins, risk_inspect_wins, tran_cost, reverse_factor, risk_factor, q_value, L_percentile);

            % Get final cumulative wealth (last value in the cum_wealth array)
            final_wealth = cum_wealth(end);

            % Store results in CSV data structure (replace hyphens with underscores for field names)
            field_name = strrep(dataset_name, '-', '_');
            csv_data.(field_name) = final_wealth;

            % Save results
            results_filename = sprintf('results/results_%s.mat', dataset_name);
            save(results_filename, 'cum_wealth', 'daily_incre_fact', 'b_history');

            fprintf('Successfully processed %s - Final cumulative wealth: %.4f\n', dataset_name, final_wealth);

            % Generate and save individual plot
            figure('Visible', 'off'); % Create a figure that is not displayed
            plot(cum_wealth, 'DisplayName', dataset_name);
            title(sprintf('Cumulative Wealth for %s', dataset_name));
            xlabel('Time Periods');
            ylabel('Cumulative Wealth');
            legend('show', 'Location', 'northwest');
            grid on;

            plot_filename = sprintf('results/figures/cumulative_wealth_%s.png', dataset_name);
            saveas(gcf, plot_filename);
            close(gcf); % Close the figure to free up memory
            fprintf('Plot saved to %s\n', plot_filename);
        catch ME
            fprintf('Error processing %s: %s\n', dataset_name, ME.message);
            field_name = strrep(dataset_name, '-', '_');
            csv_data.(field_name) = NaN; % Store NaN for failed datasets
        end

    end

    % Generate CSV file with all results
    fprintf('\nGenerating summary CSV file...\n');

    % Create CSV data row
    csv_row = zeros(1, length(csv_header));

    % Map dataset results to CSV columns
    dataset_to_column = containers.Map();
    dataset_to_column('inv30-A') = 1;
    dataset_to_column('inv30-B') = 2;
    dataset_to_column('inv30-C') = 3;
    dataset_to_column('inv500-A') = 4;
    dataset_to_column('inv500-B') = 5;
    dataset_to_column('inv500-C') = 6;
    dataset_to_column('NDX-A') = 7;
    dataset_to_column('NDX-B') = 8;
    dataset_to_column('NDX-C') = 9;

    % Fill in dataset results
    for i = 1:length(datasets)
        dataset_path = datasets{i};
        dataset_name = strrep(dataset_path, 'Data Set/', '');
        dataset_name = strrep(dataset_name, '.mat', '');

        if isKey(dataset_to_column, dataset_name)
            field_name = strrep(dataset_name, '-', '_');
            csv_row(dataset_to_column(dataset_name)) = csv_data.(field_name);
        end

    end

    % Add parameter values
    csv_row(10) = weight_inspect_wins; % weight_inspect_wins
    csv_row(11) = risk_inspect_wins; % risk_inspect_wins
    csv_row(12) = tran_cost; % tran_cost
    csv_row(13) = reverse_factor; % reverse_factor
    csv_row(14) = risk_factor; % risk_factor
    csv_row(15) = q_value; % q_value

    % Write CSV file
    csv_filename = 'results/batch_summary.csv';
    fid = fopen(csv_filename, 'w');

    if fid == -1
        error('Could not open file %s for writing', csv_filename);
    end

    % Write header
    fprintf(fid, '%s', csv_header{1});

    for j = 2:length(csv_header)
        fprintf(fid, ',%s', csv_header{j});
    end

    fprintf(fid, '\n');

    % Write data row - first 9 columns (datasets) with 10 decimal places, rest with 4 decimal places
    fprintf(fid, '%.10f', csv_row(1));

    for j = 2:9
        fprintf(fid, ',%.10f', csv_row(j));
    end

    for j = 10:length(csv_row)
        fprintf(fid, ',%.4f', csv_row(j));
    end

    fprintf(fid, '\n');

    fclose(fid);

    fprintf('Summary CSV file saved to %s\n', csv_filename);
    fprintf('Batch processing completed.\n');
end
