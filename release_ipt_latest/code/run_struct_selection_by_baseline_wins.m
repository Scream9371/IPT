function results = run_struct_selection_by_baseline_wins(varargin)
    base_dir = fileparts(mfilename('fullpath'));
    addpath(base_dir);

    p_in = inputParser;
    addParameter(p_in, 'dataset_names', {'djia', 'inv500', 'marpd', 'msci', 'nyse-n', 'nyse-o', 'sz50', 'tse'});
    addParameter(p_in, 'data_dir', fullfile(base_dir, '..', '..', 'Data Set'));
    addParameter(p_in, 'baseline_dir', fullfile(base_dir, '..', '..', 'baselines'));
    addParameter(p_in, 'results_root', fullfile(base_dir, '..', 'results_struct_selection_by_baseline_wins'));
    addParameter(p_in, 'grid_win', [126]);
    addParameter(p_in, 'grid_q', [0.1, 0.2, 0.3, 0.4, 0.5]);
    addParameter(p_in, 'grid_risk', [5, 20]);
    parse(p_in, varargin{:});
    P_in = p_in.Results;

    dataset_names = P_in.dataset_names;
    data_dir = P_in.data_dir;
    baseline_dir = P_in.baseline_dir;
    results_root = P_in.results_root;
    if ~exist(results_root, 'dir'), mkdir(results_root); end

    grid_win = P_in.grid_win;
    grid_q = P_in.grid_q;
    grid_risk = P_in.grid_risk;

    p_base = struct();
    p_base.epsilon = 100;
    p_base.tran_cost = 0.001;
    p_base.win_size = 5;
    p_base.max_turnover = Inf;
    p_base.adaptive_inertia_q = 0;
    p_base.L_smoothing_alpha = 0.2;
    p_base.Q_smoothing_alpha = 0;
    p_base.risk_inspect_wins = 21;
    p_base.L_percentile = 95;
    p_base.force_no_orth = false;
    p_base.mix = 0.5;
    p_base.clip = Inf;
    p_base.near_risk_mode = 'by_weight';

    structures = {};
    structures{end + 1} = struct('name', 'S1_OrthOn_NoQclip_Inertia', 'p', setfield(p_base, 'force_no_orth', false));
    structures{end + 1} = struct('name', 'S2_OrthOff_NoQclip_Inertia', 'p', setfield(p_base, 'force_no_orth', true));

    p = p_base; p.clip = 10; p.force_no_orth = false;
    structures{end + 1} = struct('name', 'S3_OrthOn_Qclip10_Inertia', 'p', p);
    p = p_base; p.clip = 10; p.force_no_orth = true;
    structures{end + 1} = struct('name', 'S4_OrthOff_Qclip10_Inertia', 'p', p);

    p = p_base; p.max_turnover = 0.5; p.force_no_orth = false;
    structures{end + 1} = struct('name', 'S5_OrthOn_NoQclip_Inertia_HardCap05', 'p', p);

    p = p_base; p.adaptive_inertia_q = 1; p.force_no_orth = false;
    structures{end + 1} = struct('name', 'S6_OrthOn_NoQclip_ADC', 'p', p);

    p = p_base; p.near_risk_mode = 'by_risk'; p.force_no_orth = false;
    structures{end + 1} = struct('name', 'S7_OrthOn_NoQclip_NearRiskByRisk', 'p', p);

    p = p_base; p.L_smoothing_alpha = 0; p.force_no_orth = false;
    structures{end + 1} = struct('name', 'S8_OrthOn_NoQclip_NoLsmooth', 'p', p);

    baseline_algs = {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt'};

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

            split_idx = floor(T * 0.6);
            test_idx = split_idx + 1;

            val_len = floor(split_idx * 0.2);
            val_start = split_idx - val_len + 1;
            val_end = split_idx;

            best_score = -Inf;
            best = s.p;

            for w = grid_win
                dev_data = data(1:split_idx, :);

                if w > size(dev_data, 1)
                    continue;
                end

                ratio = ubah_price_ratio(dev_data);
                yar_weights_long = yar_weights(dev_data, w);
                yar_weights_near = yar_weights(dev_data, floor(w / 2));

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
                            dev_data, w, ...
                            risk, q, L_history);

                        Q_factor = Q_factor_raw;

                        if s.p.Q_smoothing_alpha > 0
                            Q_factor = ipt_smooth_series(Q_factor, s.p.Q_smoothing_alpha);
                        end

                        if ~isinf(s.p.clip)
                            Q_factor = max(min(Q_factor, s.p.clip), -s.p.clip);
                        end

                        [~, daily_ret_val] = ipt_run_core(dev_data, s.p.win_size, s.p.tran_cost, ...
                            w_YAR, Q_factor, s.p.epsilon, s.p.mix, s.p.max_turnover, s.p.adaptive_inertia_q, s.p.force_no_orth);

                        val_end_use = min(val_end, length(daily_ret_val));
                        val_ret = daily_ret_val(val_start:val_end_use);
                        score = prod(val_ret);

                        if score > best_score
                            best_score = score;
                            best.win = w;
                            best.q = q;
                            best.risk = risk;
                        end

                    end

                end

            end

            ratio = ubah_price_ratio(data);
            yar_weights_long = yar_weights(data, best.win);
            yar_weights_near = yar_weights(data, floor(best.win / 2));

            r = best.risk_inspect_wins;
            r3 = max(2, floor(r / 3));
            start_long = best.win - r3 + 1;
            yar_ubah_long = yar_ubah(ratio(start_long:end, :), r3);

            half_r3 = max(2, floor(floor(r / 2) / 3));
            half_weight = floor(best.win / 2);
            start_near = half_weight - half_r3 + 1;

            if strcmpi(best.near_risk_mode, 'by_risk')
                yar_ubah_near = yar_ubah(ratio, half_r3);
            else
                yar_ubah_near = yar_ubah(ratio(start_near:end, :), half_r3);
            end

            L_raw = compute_yar_percentile(yar_ubah_long(:, 1), best.L_percentile);
            L_history = ipt_smooth_series(L_raw, best.L_smoothing_alpha);

            [w_YAR, Q_factor_raw] = active_function( ...
                yar_weights_long, yar_weights_near, ...
                yar_ubah_long, yar_ubah_near, ...
                data, best.win, ...
                best.risk, best.q, L_history);

            Q_factor = Q_factor_raw;

            if best.Q_smoothing_alpha > 0
                Q_factor = ipt_smooth_series(Q_factor, best.Q_smoothing_alpha);
            end

            if ~isinf(best.clip)
                Q_factor = max(min(Q_factor, best.clip), -best.clip);
            end

            [~, daily_ret] = ipt_run_core(data, best.win_size, best.tran_cost, ...
                w_YAR, Q_factor, best.epsilon, best.mix, best.max_turnover, best.adaptive_inertia_q, best.force_no_orth);

            test_ret = daily_ret(test_idx:end);
            cw = prod(test_ret);
            test_cw(si, di) = cw;

            base_key = strrep(dname, '-', '_');
            wins(si, di) = sum(cw > baseline_cw.(base_key));

            save(fullfile(sdir, ['ipt_' s.name '-' dname '.mat']), 'test_ret', 'cw', 'best', 'best_score');
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
