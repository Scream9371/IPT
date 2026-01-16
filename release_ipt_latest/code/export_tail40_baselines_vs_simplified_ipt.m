function export_tail40_baselines_vs_simplified_ipt()
    algorithms = {'UBAH', 'BCRP', 'UP', 'OLMAR2', 'RMR', 'Anticor', 'CORN', 'PPT', 'TPPT', 'iptNoClip'};
    datasets = {'djia', 'hs300', 'inv500', 'marpd', 'msci', 'ndx', 'nyse-n', 'nyse-o', 'sz50', 'tse'};
    baseline_dir = 'results_tail40_raw';
    num_alg = numel(algorithms);
    num_ds = numel(datasets);
    cum_mat = NaN(num_ds, num_alg);
    sharpe_mat = NaN(num_ds, num_alg);

    for i = 1:num_alg
        algo = algorithms{i};

        for j = 1:num_ds
            ds = datasets{j};

            if strcmp(algo, 'iptNoClip')
                filename = sprintf('iptNoClip-%s_tail40.mat', ds);
            else
                filename = sprintf('%s-%s_tail40.mat', lower(algo), ds);
            end

            filepath = fullfile(baseline_dir, filename);

            if exist(filepath, 'file')
                S = load(filepath);

                if isfield(S, 'cum_ret')
                    cum_mat(j, i) = double(S.cum_ret);
                elseif isfield(S, 'daily_ret')
                    x = double(S.daily_ret(:));

                    if ~isempty(x) && all(isfinite(x)) && all(x > 0)
                        cum_mat(j, i) = prod(x);
                    end

                end

                if isfield(S, 'daily_ret')
                    seg = double(S.daily_ret(:));

                    if numel(seg) >= 2 && all(isfinite(seg)) && all(seg > 0)
                        r = seg - 1;
                        s = std(r, 0);

                        if isfinite(s) && s > 0
                            sharpe_mat(j, i) = sqrt(252) * (mean(r) / s);
                        end

                    end

                end

            end

        end

    end

    out_ds = strings(num_ds, 1);

    for j = 1:num_ds
        out_ds(j) = datasets{j};
    end

    var_names = cell(1, 1 + 2 * num_alg);
    cols = cell(1, 1 + 2 * num_alg);
    var_names{1} = 'dataset';
    cols{1} = out_ds;
    c = 2;

    for i = 1:num_alg
        var_names{c} = [algorithms{i} '_cum_ret'];
        cols{c} = cum_mat(:, i);
        c = c + 1;
        var_names{c} = [algorithms{i} '_sharpe'];
        cols{c} = sharpe_mat(:, i);
        c = c + 1;
    end

    Tout = table(cols{:}, 'VariableNames', var_names);
    out_csv = fullfile('results_tail40_raw', 'tail40_unclipped_all_vs_iptNoClip.csv');
    writetable(Tout, out_csv);
    disp(out_csv);
end
