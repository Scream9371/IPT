function out = ipt_paper_tables(varargin)
    %IPT_PAPER_TABLES  Generate paper.tex tables for the 8-dataset setting.
    %
    % This helper reads a results folder (with *_tail40.mat files and stat_*.csv)
    % and exports ready-to-paste LaTeX table bodies for:
    %   - Table 7  (CW)
    %   - Table 8  (APY)
    %   - Table 9  (Sharpe)
    %   - Table 10 (Calmar)
    %   - Table 11 (MER + one-sided t-test p-values vs UBAH)
    %   - Table 12 (Winning rate of IPT vs UBAH, per dataset)
    %   - Table 16/19 (Friedman test summaries)
    %   - Table 17/18 (Ranking tables)
    %
    % Usage:
    %   out = ipt_paper_tables('results_dir', fullfile(pwd,'results_paper10_tail40_noTurnover_plus_djia'));
    %
    % Outputs:
    %   - out (struct) with numeric matrices and LaTeX strings
    %   - Writes a snippet file: <results_dir>/paper_tables_plus_djia_snippet.tex
    %
    p = inputParser;
    addParameter(p, 'results_dir', fullfile(fileparts(mfilename('fullpath')), 'results_paper10_tail40_noTurnover_plus_djia'));
    addParameter(p, 'baseline_dir', '');
    addParameter(p, 'datasets_order', {'djia', 'inv500', 'msci', 'nyse-n', 'nyse-o', 'sz50', 'tse', 'marpd'});
    addParameter(p, 'algos_order', {'ubah', 'bcrp', 'up', 'olmar2', 'rmr', 'anticor', 'corn', 'ppt', 'tppt', 'ipt_best_noqclip'});
    addParameter(p, 'algo_display', {'UBAH', 'BCRP', 'UP', 'OLMAR-2', 'RMR', 'Anticor', 'CORN', 'PPT', 'TPPT', 'IPT'});
    addParameter(p, 'annualization', 252);
    addParameter(p, 'alpha', 0.05);
    parse(p, varargin{:});
    opts = p.Results;

    results_dir = char(string(opts.results_dir));
    baseline_dir = char(string(opts.baseline_dir));

    if ~exist(results_dir, 'dir')
        error('results_dir not found: %s', results_dir);
    end

    datasets = lower(string(opts.datasets_order(:)'));
    algos = lower(string(opts.algos_order(:)'));
    algo_display = string(opts.algo_display(:)');
    mask_no_inv500 = true(1, numel(datasets));

    Tcw = readtable(fullfile(results_dir, 'stat_terminal_cumwealth.csv'));
    Tsr = readtable(fullfile(results_dir, 'stat_sharpe.csv'));
    Tcw.dataset = lower(string(Tcw.dataset));
    Tsr.dataset = lower(string(Tsr.dataset));

    assert(ismember("dataset", string(Tcw.Properties.VariableNames)));
    assert(ismember("dataset", string(Tsr.Properties.VariableNames)));

    for a = algos
        assert(ismember(char(a), Tcw.Properties.VariableNames), 'Missing algo=%s in stat_terminal_cumwealth.csv', char(a));
        assert(ismember(char(a), Tsr.Properties.VariableNames), 'Missing algo=%s in stat_sharpe.csv', char(a));
    end

    for d = datasets
        assert(any(Tcw.dataset == d), 'Missing dataset=%s in stat_terminal_cumwealth.csv', char(d));
        assert(any(Tsr.dataset == d), 'Missing dataset=%s in stat_sharpe.csv', char(d));
    end

    cw = nan(numel(algos), numel(datasets));
    sr = nan(numel(algos), numel(datasets));
    n_days = nan(1, numel(datasets));
    apy = nan(numel(algos), numel(datasets));
    calmar = nan(numel(algos), numel(datasets));

    % Load a single file per (algo,dataset) only when needed.
    for j = 1:numel(datasets)
        d = datasets(j);

        ubah_path = find_result_file(sprintf('ubah-%s_tail40.mat', d), results_dir, baseline_dir);

        if ~isfile(ubah_path)
            error('Missing file needed to infer tail length: %s', ubah_path);
        end

        Su = load_series_vars(ubah_path);
        ru = extract_factor_series(Su);
        n_days(j) = numel(ru);

        if n_days(j) < 2
            error('Too few points in %s (n=%d)', ubah_path, n_days(j));
        end

    end

    for i = 1:numel(algos)
        a = algos(i);

        for j = 1:numel(datasets)
            d = datasets(j);
            cw(i, j) = double(Tcw{Tcw.dataset == d, char(a)});
            sr(i, j) = double(Tsr{Tsr.dataset == d, char(a)});

            n = n_days(j);
            apy(i, j) = cw(i, j) ^ (opts.annualization / n) - 1;

            file_path = find_result_file(sprintf('%s-%s_tail40.mat', a, d), results_dir, baseline_dir);

            if ~isfile(file_path)
                error('Missing .mat result: %s', file_path);
            end

            S = load_series_vars(file_path);
            curve = extract_curve(S);
            mdd = max_drawdown(curve);

            if mdd > 0
                calmar(i, j) = apy(i, j) / mdd;
            else
                calmar(i, j) = Inf;
            end

        end

    end

    % MER & one-sided t-test p-values: compare daily returns vs UBAH as market.
    % Use all algorithms except UBAH itself, preserving the order in algos.
    mer_algos = algos(algos ~= "ubah");
    mer_display = string({'BCRP', 'UP', 'OLMAR', 'RMR', 'Anticor', 'CORN', 'PPT', 'TPPT', 'IPT'});
    mer = nan(numel(mer_algos), numel(datasets));
    mer_p = nan(numel(mer_algos), numel(datasets));

    for j = 1:numel(datasets)
        d = datasets(j);

        ubah_path = find_result_file(sprintf('ubah-%s_tail40.mat', d), results_dir, baseline_dir);
        Su = load_series_vars(ubah_path);
        fac_m = extract_factor_series(Su);
        rm = fac_m(:) - 1;

        for i = 1:numel(mer_algos)
            a = mer_algos(i);

            sx_path = find_result_file(sprintf('%s-%s_tail40.mat', a, d), results_dir, baseline_dir);
            Sx = load_series_vars(sx_path);
            fac_s = extract_factor_series(Sx);
            rs = fac_s(:) - 1;

            n = min(numel(rs), numel(rm));
            ex = rs(1:n) - rm(1:n);
            ex = ex(isfinite(ex));
            mer(i, j) = mean(ex);

            if numel(ex) >= 5
                [~, pval] = ttest(ex, 0, 'Tail', 'right');
                mer_p(i, j) = pval;
            else
                mer_p(i, j) = NaN;
            end

        end

    end

    % Winning rate of IPT-like target vs UBAH: P(excess > 0) per dataset.
    ipt_algo = algos(end);
    winrate = nan(1, numel(datasets));

    for j = 1:numel(datasets)
        d = datasets(j);

        ubah_path = find_result_file(sprintf('ubah-%s_tail40.mat', d), results_dir, baseline_dir);
        Su = load_series_vars(ubah_path);

        ipt_path = find_result_file(sprintf('%s-%s_tail40.mat', ipt_algo, d), results_dir, baseline_dir);
        Si = load_series_vars(ipt_path);

        fac_m = extract_factor_series(Su);
        fac_s = extract_factor_series(Si);
        rm = fac_m(:) - 1;
        rs = fac_s(:) - 1;
        n = min(numel(rs), numel(rm));
        ex = rs(1:n) - rm(1:n);
        winrate(j) = mean(ex > 0);
    end

    % Ranking tables and Friedman test (Iman-Davenport) stats.
    cw_rank = cw(:, mask_no_inv500);
    sr_rank = sr(:, mask_no_inv500);
    ranks_cw = rank_matrix_desc(cw_rank);
    ranks_sr = rank_matrix_desc(sr_rank);
    [fried_cw, CD_cw] = friedman_summary(ranks_cw, opts.alpha);
    [fried_sr, CD_sr] = friedman_summary(ranks_sr, opts.alpha);

    % Build LaTeX snippets.
    ds_header = upper(strrep(datasets, "-", "-"));
    % Use "NYSE(N)" / "NYSE(O)" display like the paper.
    ds_header(ds_header == "NYSE-N") = "NYSE(N)";
    ds_header(ds_header == "NYSE-O") = "NYSE(O)";

    out = struct();
    out.meta = struct('results_dir', results_dir, 'datasets', datasets, 'algos', algos);
    out.cw = cw; out.sr = sr; out.apy = apy; out.calmar = calmar;
    out.mer_algos = mer_algos; out.mer = mer; out.mer_p = mer_p;
    out.winrate = winrate;
    out.ranks_cw = ranks_cw; out.ranks_sr = ranks_sr;
    out.friedman_cw = fried_cw; out.friedman_sr = fried_sr;
    out.CD_cw = CD_cw; out.CD_sr = CD_sr;

    out.tex.table7_rows = build_rows_1metric(algo_display, cw, 'cw');
    out.tex.table8_rows = build_rows_1metric(algo_display, apy, 'apy');
    out.tex.table9_rows = build_rows_1metric(algo_display, sr, 'sr');
    out.tex.table10_rows = build_rows_1metric(algo_display, calmar(:, mask_no_inv500), 'calmar');
    out.tex.table11_rows = build_rows_mer(mer_display, mer(:, mask_no_inv500), mer_p(:, mask_no_inv500));
    out.tex.table12_row = build_row_winrate(winrate);
    out.tex.table16 = build_friedman_table(fried_cw);
    out.tex.table19 = build_friedman_table(fried_sr);
    out.tex.table17_rows = build_rows_ranks(algo_display, ranks_cw);
    out.tex.table18_rows = build_rows_ranks(algo_display, ranks_sr);
    out.tex.datasets_header = ds_header;

    snippet_path = fullfile(results_dir, 'paper_tables_plus_djia_snippet.tex');
    fid = fopen(snippet_path, 'w');

    if fid == -1
        error('Cannot write snippet: %s', snippet_path);
    end

    c = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%% Auto-generated from: %s\n\n', results_dir);
    fprintf(fid, '%% Datasets: %s\n', strjoin(ds_header, ', '));
    fprintf(fid, '\n%% Table 7 rows (CW)\n%s\n', out.tex.table7_rows);
    fprintf(fid, '\n%% Table 8 rows (APY)\n%s\n', out.tex.table8_rows);
    fprintf(fid, '\n%% Table 9 rows (Sharpe)\n%s\n', out.tex.table9_rows);
    fprintf(fid, '\n%% Table 10 rows (Calmar)\n%s\n', out.tex.table10_rows);
    fprintf(fid, '\n%% Table 11 rows (MER + p)\n%s\n', out.tex.table11_rows);
    fprintf(fid, '\n%% Table 12 row (Winning rate of IPT)\n%s\n', out.tex.table12_row);
    fprintf(fid, '\n%% Table 16 (Friedman CW)\n%s\n', out.tex.table16);
    fprintf(fid, '\n%% Table 17 rows (Rank CW)\n%s\n', out.tex.table17_rows);
    fprintf(fid, '\n%% Table 18 rows (Rank Sharpe)\n%s\n', out.tex.table18_rows);
    fprintf(fid, '\n%% Table 19 (Friedman Sharpe)\n%s\n', out.tex.table19);

    root_dir = fileparts(fileparts(mfilename('fullpath')));
    paper_dir = fullfile(root_dir, 'docs', 'paper_tables');

    if ~exist(paper_dir, 'dir')
        mkdir(paper_dir);
    end

    minimal_path = fullfile(paper_dir, 'paper_tables_minimal.tex');

    if ~isfile(minimal_path)

        try
            copyfile(snippet_path, minimal_path);
        catch
        end

    end

end

function S = load_series_vars(file_path)
    % Load only series variables we may use; avoid warnings from missing vars.
    S = load(file_path, '-regexp', '^(daily_ret|daily_incre_fact|cumprod_ret|cum_wealth|cumwealth|cum_wealth_curve|cum_ret)$');

    if isempty(fieldnames(S))
        S = load(file_path);
    end

end

function fac = extract_factor_series(S)

    if isfield(S, 'daily_incre_fact') && isnumeric(S.daily_incre_fact) && ~isempty(S.daily_incre_fact)
        fac = double(S.daily_incre_fact(:));
        return;
    end

    if isfield(S, 'daily_ret') && isnumeric(S.daily_ret) && ~isempty(S.daily_ret)
        x = double(S.daily_ret(:));

        if median(x) > 0.5 && all(x > 0)
            fac = x;
        else
            fac = 1 + x;
        end

        return;
    end

    error('No daily return series found in loaded struct.');
end

function curve = extract_curve(S)

    if isfield(S, 'cumprod_ret') && isnumeric(S.cumprod_ret) && ~isempty(S.cumprod_ret)
        curve = double(S.cumprod_ret(:));
        return;
    end

    if isfield(S, 'cum_wealth') && isnumeric(S.cum_wealth) && ~isempty(S.cum_wealth)
        curve = double(S.cum_wealth(:));
        return;
    end

    if isfield(S, 'cumwealth') && isnumeric(S.cumwealth) && ~isempty(S.cumwealth)
        curve = double(S.cumwealth(:));
        return;
    end

    if isfield(S, 'cum_wealth_curve') && isnumeric(S.cum_wealth_curve) && ~isempty(S.cum_wealth_curve)
        curve = double(S.cum_wealth_curve(:));
        return;
    end

    fac = extract_factor_series(S);
    curve = cumprod(fac(:));
end

function mdd = max_drawdown(curve)
    c = double(curve(:));
    c = c(isfinite(c) & c > 0);

    if numel(c) < 2
        mdd = NaN;
        return;
    end

    peak = c(1);
    mdd = 0;

    for i = 2:numel(c)

        if c(i) > peak
            peak = c(i);
        else
            dd = 1 - c(i) / peak;

            if dd > mdd
                mdd = dd;
            end

        end

    end

end

function ranks = rank_matrix_desc(metric)
    % metric: (algos x datasets)
    [k, N] = size(metric);
    ranks = nan(k, N);

    for j = 1:N
        ranks(:, j) = tiedrank(-metric(:, j)); % rank 1 = best
    end

end

function [fried, CD] = friedman_summary(ranks, alpha)
    % ranks: (k x N) but compute per dataset => transpose to (N x k)
    k = size(ranks, 1);
    N = size(ranks, 2);
    ranksN = ranks'; % (N x k)
    sum_ranks = sum(ranksN, 1);
    mean_ranks = sum_ranks / N;
    chi2 = (12 / (N * k * (k + 1))) * sum(sum_ranks .^ 2) - 3 * N * (k + 1);
    df1 = k - 1;
    Ff = ((N - 1) * chi2) / (N * df1 - chi2);
    df2 = df1 * (N - 1);
    pF = 1 - fcdf(Ff, df1, df2);
    Fcrit = finv(1 - alpha, df1, df2);
    q_alpha = 3.164; % for k=10 and alpha=0.05 (Demsar 2006); matches run_statistical_tests default table
    CD = q_alpha * sqrt(k * (k + 1) / (6 * N));
    fried = struct('N', N, 'k', k, 'alpha', alpha, 'chi2', chi2, 'F_f', Ff, 'p', pF, 'F_crit', Fcrit, 'mean_ranks', mean_ranks);
end

function s = build_rows_1metric(algo_display, M, kind)
    parts = strings(0);

    for i = 1:size(M, 1)
        row = algo_display(i) + " & " + strjoin(format_vals(M(i, :), kind), " & ") + " \\\\";
        parts(end + 1) = row; %#ok<AGROW>
    end

    s = strjoin(parts, newline);
end

function s = build_rows_mer(model_display, mer, p)
    % Builds rows like: Model & MER & Pvalue & MER & Pvalue ...
    parts = strings(0);

    for i = 1:size(mer, 1)
        cells = strings(1, size(mer, 2) * 2);

        for j = 1:size(mer, 2)
            cells(2 * j - 1) = format_number(mer(i, j), 'mer');
            cells(2 * j) = format_pvalue(p(i, j));
        end

        row = model_display(i) + " & " + strjoin(cells, " & ") + " \\\\";
        parts(end + 1) = row; %#ok<AGROW>
    end

    s = strjoin(parts, newline);
end

function s = build_row_winrate(winrate)
    s = "IPT & " + strjoin(format_vals(winrate, 'winrate'), " & ") + " \\\\";
end

function s = build_friedman_table(fried)
    s = sprintf([ ...
                     'N & $%d$ \\\\\n' ...
                     'k & $%d$\\\\\n' ...
                     '$\\alpha$ & $%.2f$\\\\\n' ...
                     '$\\chi^2$ & $%.2f$\\\\\n' ...
                     '$F_f$ & $%.2f$\\\\\n' ...
                     '$p$-value & $%.4g$\\\\\n' ...
                     '$F_{\\alpha}(k-1,(N-1)(k-1))$ & $%.2f$\\\\' ...
                 ], ...
        fried.N, fried.k, fried.alpha, fried.chi2, fried.F_f, fried.p, fried.F_crit);
end

function s = build_rows_ranks(algo_display, ranks)
    avg_rank = mean(ranks, 2);
    best = min(avg_rank);
    diff = avg_rank - best;
    parts = strings(0);

    for i = 1:size(ranks, 1)
        row = algo_display(i) + " & " + strjoin(format_vals(ranks(i, :), 'rank'), " & ") + ...
            " & " + format_number(avg_rank(i), 'rank_avg') + " & " + format_number(diff(i), 'rank_avg') + "\\\\";
        parts(end + 1) = row; %#ok<AGROW>
    end

    s = strjoin(parts, newline);
end

function vals = format_vals(x, kind)
    vals = strings(1, numel(x));

    for i = 1:numel(x)
        vals(i) = format_number(x(i), kind);
    end

end

function s = format_pvalue(p)

    if ~isfinite(p)
        s = "NA";
        return;
    end

    if p < 1e-4
        s = "$<0.0001$";
        return;
    end

    s = sprintf('%.4f', p);
end

function s = format_number(v, kind)

    if ~isfinite(v)
        s = "NA";
        return;
    end

    if kind == "rank"

        if abs(v - round(v)) < 1e-9
            s = sprintf('%d', round(v));
        else
            s = sprintf('%.1f', v);
        end

        return;
    end

    if kind == "rank_avg"
        s = sprintf('%.2f', v);
        return;
    end

    if kind == "winrate"
        s = sprintf('%.4f', v);
        return;
    end

    if kind == "mer"
        s = sprintf('%.4f', v);

        if startsWith(s, '-0.0000')
            s = '-0.0000';
        end

        return;
    end

    % CW / APY / Sharpe / Calmar: keep the paper's style (scientific for huge).
    abs_v = abs(v);

    if abs_v >= 1e4
        [mant, exp10] = sci_parts(v);
        s = sprintf('$%.3f \\times 10^{%d}$', mant, exp10);
        return;
    end

    if kind == "apy" || kind == "sr" || kind == "calmar"
        s = sprintf('%.4f', v);
        return;
    end

    % Default CW.
    s = sprintf('%.4f', v);
end

function [mant, exp10] = sci_parts(v)
    exp10 = floor(log10(abs(v)));
    mant = v / (10 ^ exp10);
end

function path = find_result_file(filename, results_dir, baseline_dir)
    path = fullfile(results_dir, filename);

    if isfile(path)
        return;
    end

    if ~isempty(baseline_dir)
        path_base = fullfile(baseline_dir, filename);

        if isfile(path_base)
            path = path_base;
            return;
        end

    end

    % If neither exists, let the caller handle the missing file (path points to results_dir).
end
