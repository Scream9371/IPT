function results = run_ipt(varargin)
    base_dir = fileparts(mfilename('fullpath'));
    addpath(base_dir);

    p_in = inputParser;
    addParameter(p_in, 'mode', 'struct_selection');
    addParameter(p_in, 'dataset_names', {'djia', 'inv500', 'marpd', 'msci', 'nyse-n', 'nyse-o', 'sz50', 'tse'});
    addParameter(p_in, 'data_dir', fullfile(base_dir, '..', '..', 'Data Set'));
    addParameter(p_in, 'baseline_dir', fullfile(base_dir, '..', '..', 'baselines'));
    addParameter(p_in, 'results_root', fullfile(base_dir, '..', 'results_struct_selection_by_baseline_wins'));
    addParameter(p_in, 'grid_win', [126]);
    addParameter(p_in, 'grid_q', [0.1, 0.2, 0.3, 0.4, 0.5]);
    addParameter(p_in, 'grid_risk', [5, 20]);
    addParameter(p_in, 'baseline_algs', {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt'});
    addParameter(p_in, 'p_base', []);
    addParameter(p_in, 'p_base_overrides', struct());
    addParameter(p_in, 'structures', {});
    addParameter(p_in, 'structure_specs', {});
    addParameter(p_in, 'run_tag', '');
    addParameter(p_in, 'save_per_dataset_mat', true);
    addParameter(p_in, 'val_metric', 'wealth'); % 'wealth' | 'log_wealth'
    addParameter(p_in, 'K', 5);
    parse(p_in, varargin{:});
    P_in = p_in.Results;

    mode = lower(string(P_in.mode));

    if mode ~= "struct_selection"
        error('Unsupported mode: %s (use struct_selection)', mode);
    end

    dataset_names = P_in.dataset_names;
    data_dir = P_in.data_dir;
    baseline_dir = P_in.baseline_dir;
    results_root = P_in.results_root;
    if ~exist(results_root, 'dir'), mkdir(results_root); end

    grid_win = P_in.grid_win;
    grid_q = P_in.grid_q;
    grid_risk = P_in.grid_risk;
    baseline_algs = P_in.baseline_algs;

    val_metric = lower(string(P_in.val_metric));

    if val_metric ~= "wealth" && val_metric ~= "log_wealth"
        error('Unsupported val_metric: %s (use wealth or log_wealth)', val_metric);
    end

    if isempty(P_in.p_base)
        p_base = local_default_p_base();
    else
        p_base = P_in.p_base;
    end

    p_base = local_struct_override(p_base, P_in.p_base_overrides);

    if ~isempty(P_in.structures)
        structures = P_in.structures;
    elseif ~isempty(P_in.structure_specs)
        structures = local_parse_structure_specs(P_in.structure_specs, p_base);
    else
        structures = local_default_structures(p_base);
    end

    baseline_cw = struct();

    for di = 1:numel(dataset_names)
        dname = dataset_names{di};
        cws = nan(1, numel(baseline_algs));

        for ai = 1:numel(baseline_algs)
            f = fullfile(baseline_dir, [baseline_algs{ai} '-' dname '_tail40.mat']);
            S = load(f);

            if isfield(S, 'cumprod_ret')
                cws(ai) = S.cumprod_ret(end);
            elseif isfield(S, 'daily_ret')
                cws(ai) = prod(S.daily_ret);
            else
                error('Baseline file missing cumprod_ret/daily_ret: %s', f);
            end

        end

        baseline_cw.(strrep(dname, '-', '_')) = cws;
    end

    num_struct = numel(structures);
    num_data = numel(dataset_names);
    wins = zeros(num_struct, num_data);
    test_cw = nan(num_struct, num_data);

    fprintf('=== Struct Selection (criterion: wins vs 9 baselines, CW higher) ===\n');
    fprintf('Grid: win=%s, q=%s, risk=%s | cost=%.4f\n', mat2str(grid_win), mat2str(grid_q), mat2str(grid_risk), p_base.tran_cost);

    for si = 1:num_struct
        s = structures{si};

        if ~isfield(s, 'name') || ~isfield(s, 'p')
            error('Invalid structures{%d}: require fields name and p', si);
        end

        s.p = local_struct_defaults(s.p, p_base);
        sdir = fullfile(results_root, s.name);
        if ~exist(sdir, 'dir'), mkdir(sdir); end

        fprintf('\n--- %s ---\n', s.name);

        for di = 1:numel(dataset_names)
            dname = dataset_names{di};
            data_path = fullfile(data_dir, [dname '.mat']);

            if ~isfile(data_path)
                continue;
            end

            Sdata = load(data_path, 'data');
            data = Sdata.data;
            T = size(data, 1);

            dev_end = floor(T * 0.6);
            test_start = dev_end + 1;
            test_end = T;

            warmup_end = max([max(grid_win), s.p.risk_inspect_wins, s.p.win_size]);
            tune_start = warmup_end + 1;
            tune_end = dev_end;

            if tune_start > tune_end || test_start > test_end
                continue;
            end

            K = max(1, round(double(P_in.K)));
            val_len_total = tune_end - tune_start + 1;
            K = min(K, val_len_total);
            fold_len = floor(val_len_total / K);

            if fold_len < 1
                K = 1;
                fold_len = val_len_total;
            end

            fold_ranges = zeros(K, 2);

            for k = 1:K
                f_start = tune_start + (k - 1) * fold_len;

                if k == K
                    f_end = tune_end;
                else
                    f_end = f_start + fold_len - 1;
                end

                fold_ranges(k, :) = [f_start, f_end];
            end

            best_score = -Inf;
            best = s.p;
            best_daily_ret = [];

            ratio = ubah_price_ratio(data);

            for w = grid_win

                if w > T
                    continue;
                end

                yar_weights_long = yar_weights(data, w);
                yar_weights_near = yar_weights(data, floor(w / 2));
                r = s.p.risk_inspect_wins;
                r3 = max(2, floor(r / 3));
                start_long = w - r3 + 1;

                if start_long < 1
                    continue;
                end

                yar_ubah_long = yar_ubah(ratio(start_long:end, :), r3);

                half_r3 = max(2, floor(floor(r / 2) / 3));
                half_weight = floor(w / 2);
                start_near = half_weight - half_r3 + 1;

                if start_near < 1
                    continue;
                end

                if strcmpi(s.p.near_risk_mode, 'by_risk')
                    yar_ubah_near = yar_ubah(ratio, half_r3);
                else
                    yar_ubah_near = yar_ubah(ratio(start_near:end, :), half_r3);
                end

                L_raw = compute_yar_percentile(yar_ubah_long(:, 1), s.p.L_percentile);
                L_history = ipt_smooth_series(L_raw, s.p.L_smoothing_alpha);

                for q = grid_q

                    for risk = grid_risk
                        [w_YAR, Q_factor_raw] = active_function( ...
                            yar_weights_long, yar_weights_near, ...
                            yar_ubah_long, yar_ubah_near, ...
                            data, w, ...
                            risk, q, L_history);

                        Q_factor = Q_factor_raw;

                        if s.p.Q_smoothing_alpha > 0
                            Q_factor = ipt_smooth_series(Q_factor, s.p.Q_smoothing_alpha);
                        end

                        if ~isinf(s.p.clip)
                            Q_factor = max(min(Q_factor, s.p.clip), -s.p.clip);
                        end

                        [~, daily_ret_all] = ipt_run_core(data, s.p.win_size, s.p.tran_cost, ...
                            w_YAR, Q_factor, s.p.epsilon, s.p.mix, s.p.max_turnover, s.p.adaptive_inertia_q, s.p.force_no_orth);
                        fold_wealths = zeros(K, 1);
                        fold_logs = zeros(K, 1);

                        for k = 1:K
                            s_idx = fold_ranges(k, 1);
                            e_idx = fold_ranges(k, 2);
                            w_fold = prod(daily_ret_all(s_idx:e_idx));
                            fold_wealths(k) = w_fold;
                            fold_logs(k) = log(max(w_fold, realmin));
                        end

                        score_log = mean(fold_logs);

                        if val_metric == "log_wealth"
                            score = score_log;
                        else
                            score = exp(score_log);
                        end

                        if score > best_score
                            best_score = score;
                            best.win = w;
                            best.q = q;
                            best.risk = risk;
                            best_daily_ret = daily_ret_all;
                        end

                    end

                end

            end

            if isempty(best_daily_ret)
                continue;
            end

            test_ret = best_daily_ret(test_start:test_end);
            cw = prod(test_ret);
            test_cw(si, di) = cw;

            base_key = strrep(dname, '-', '_');
            wins(si, di) = sum(cw > baseline_cw.(base_key));

            if P_in.save_per_dataset_mat
                tag = char(string(P_in.run_tag));

                if isempty(tag)
                    out_name = ['ipt_' s.name '-' dname '.mat'];
                else
                    out_name = ['ipt_' s.name '-' dname '-' tag '.mat'];
                end

                split = struct('dev_end', dev_end, 'test_start', test_start, 'test_end', test_end, ...
                    'warmup_end', warmup_end, 'tune_start', tune_start, 'tune_end', tune_end, 'K', K, 'fold_ranges', fold_ranges);
                save(fullfile(sdir, out_name), 'test_ret', 'cw', 'best', 'best_score', 'split');
            end

            fprintf('  %-7s | wins=%d/9 | cw=%.4f\n', dname, wins(si, di), cw);
        end

        fprintf('  TOTAL wins: %d / %d\n', sum(wins(si, :)), num_data * numel(baseline_algs));
    end

    [best_total, best_idx] = max(sum(wins, 2));
    fprintf('\n=== Best by total wins ===\n');
    fprintf('%s | total wins=%d\n', structures{best_idx}.name, best_total);

    results = struct();
    results.wins = wins;
    results.test_cw = test_cw;
    results.structures = structures;
    results.grid = struct('win', grid_win, 'q', grid_q, 'risk', grid_risk);
    results.dataset_names = dataset_names;
    results.baseline_algs = baseline_algs;
    results.best = struct('name', structures{best_idx}.name, 'total_wins', best_total);
end

function p = local_default_p_base()
    p = struct();
    p.epsilon = 100;
    p.tran_cost = 0.001;
    p.win_size = 5;
    p.max_turnover = Inf;
    p.adaptive_inertia_q = 0;
    p.L_smoothing_alpha = 0.2;
    p.Q_smoothing_alpha = 0;
    p.risk_inspect_wins = 21;
    p.L_percentile = 95;
    p.force_no_orth = false;
    p.mix = 0.5;
    p.clip = Inf;
    p.near_risk_mode = 'by_weight';
end

function s = local_struct_override(s, overrides)

    if isempty(overrides)
        return;
    end

    f = fieldnames(overrides);

    for i = 1:numel(f)
        s.(f{i}) = overrides.(f{i});
    end

end

function s = local_struct_defaults(s, defaults)
    f = fieldnames(defaults);

    for i = 1:numel(f)
        k = f{i};

        if ~isfield(s, k)
            s.(k) = defaults.(k);
        end

    end

end

function structures = local_default_structures(p_base)
    structures = {};
    structures{end + 1} = struct('name', 'S1_OrthOn_NoQclip_Inertia', 'p', local_struct_override(p_base, struct('force_no_orth', false, 'clip', Inf)));
    structures{end + 1} = struct('name', 'S2_OrthOff_NoQclip_Inertia', 'p', local_struct_override(p_base, struct('force_no_orth', true, 'clip', Inf)));

    p = p_base; p.clip = 10; p.force_no_orth = false;
    structures{end + 1} = struct('name', 'S3_OrthOn_Qclip10_Inertia', 'p', p);
    p = p_base; p.clip = 10; p.force_no_orth = true;
    structures{end + 1} = struct('name', 'S4_OrthOff_Qclip10_Inertia', 'p', p);

    p = p_base; p.max_turnover = 0.5; p.force_no_orth = false; p.clip = Inf;
    structures{end + 1} = struct('name', 'S5_OrthOn_NoQclip_Inertia_HardCap05', 'p', p);

    p = p_base; p.adaptive_inertia_q = 1; p.force_no_orth = false; p.clip = Inf;
    structures{end + 1} = struct('name', 'S6_OrthOn_NoQclip_ADC', 'p', p);

    p = p_base; p.near_risk_mode = 'by_risk'; p.force_no_orth = false; p.clip = Inf;
    structures{end + 1} = struct('name', 'S7_OrthOn_NoQclip_NearRiskByRisk', 'p', p);

    p = p_base; p.L_smoothing_alpha = 0; p.force_no_orth = false; p.clip = Inf;
    structures{end + 1} = struct('name', 'S8_OrthOn_NoQclip_NoLsmooth', 'p', p);
end

function structures = local_parse_structure_specs(specs, p_base)

    if isstring(specs) || ischar(specs)
        specs = cellstr(string(specs));
    end

    if ~iscell(specs)
        error('structure_specs must be a string/char or a cell array of strings.');
    end

    structures = {};

    for i = 1:numel(specs)
        raw = string(specs{i});
        raw = strtrim(raw);

        if raw == ""
            continue;
        end

        name = "";
        overrides = struct();

        if contains(raw, ":")
            parts = split(raw, ":", 2);
            name = strtrim(parts(1));
            kv_str = strtrim(parts(2));
        else
            kv_str = raw;
        end

        if name == ""
            name = "spec" + string(i);
        end

        tokens = regexp(char(kv_str), '[,;]', 'split');

        for ti = 1:numel(tokens)
            t = strtrim(string(tokens{ti}));

            if t == ""
                continue;
            end

            if ~contains(t, "=")
                continue;
            end

            kv = split(t, "=", 2);
            k = strtrim(kv(1));
            v_raw = strtrim(kv(2));

            if k == ""
                continue;
            end

            if lower(k) == "name"
                name = string(v_raw);
                continue;
            end

            overrides.(char(k)) = local_parse_value(v_raw);
        end

        p = local_struct_override(p_base, overrides);
        structures{end + 1} = struct('name', char(name), 'p', p);
    end

    if isempty(structures)
        error('structure_specs is provided but no valid spec was parsed.');
    end

end

function v = local_parse_value(v_raw)
    s = lower(string(v_raw));

    if s == "true"
        v = true;
        return;
    end

    if s == "false"
        v = false;
        return;
    end

    if s == "inf"
        v = Inf;
        return;
    end

    if s == "-inf"
        v = -Inf;
        return;
    end

    num = str2double(s);

    if ~isnan(num)
        v = num;
        return;
    end

    v = char(string(v_raw));
end
