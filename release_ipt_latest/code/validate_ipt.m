function validate_ipt()
    script_dir = fileparts(mfilename('fullpath'));
    ppt_dir = fullfile(script_dir, '..', '..', '..', 'PPT');

    if ~exist(ppt_dir, 'dir')
        ppt_dir = fullfile(script_dir, '..', '..', '..', 'PPT&RPPT_two_wins');
    end

    if ~exist(ppt_dir, 'dir')
        error('Cannot find PPT implementation directory.');
    end

    addpath(ppt_dir, '-begin');
    addpath(script_dir, '-begin');

    data_dir = fullfile(script_dir, '..', '..', 'Data Set');
    data_path = fullfile(data_dir, 'msci.mat');

    if ~isfile(data_path)
        files = dir(fullfile(data_dir, '*.mat'));
        data_path = fullfile(data_dir, files(1).name);
    end

    S = load(data_path, 'data');
    data = S.data;
    [T, N] = size(data);

    win_size = 5;
    epsilon = 100;
    tran_cost = 0.001;

    [~, ~, ppt_b] = PPT_run(data, win_size, tran_cost);

    w_YAR_dummy = zeros(T, N);
    Q_factor_dummy = zeros(T, 1);
    [~, ~, ipt_b] = ipt_run_core(data, win_size, tran_cost, w_YAR_dummy, Q_factor_dummy, epsilon, 1.0, Inf);

    diff = abs(ppt_b - ipt_b);
    max_diff = max(diff(:));

    if max_diff < 1e-12
        fprintf('PASS: IPT(Q=0) matches PPT (max_diff=%.3e)\n', max_diff);
    else
        [r, c] = find(diff > 1e-12, 1);
        fprintf('FAIL: max_diff=%.3e, first divergence t=%d asset=%d (PPT=%.16g, IPT=%.16g)\n', max_diff, c, r, ppt_b(r, c), ipt_b(r, c));
    end

    rmpath(ppt_dir);
end
