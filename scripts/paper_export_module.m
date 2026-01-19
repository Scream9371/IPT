function out = paper_export_module(varargin)
    %PAPER_EXPORT_MODULE Run paper-oriented exports for IPT results.
    %
    % This module wraps export_olps_mat_results_all and writes outputs under
    % release_ipt_latest/results by default.
    %
    % Usage:
    %   paper_export_module('summary_csv', '...', 'algo_name', 'iptX');
    %
    p = inputParser;
    addParameter(p, 'summary_csv', '');
    addParameter(p, 'results_dir', fullfile(fileparts(mfilename('fullpath')), '..', 'results'));
    addParameter(p, 'export_olps', true);
    addParameter(p, 'olps_dir', '');
    addParameter(p, 'L_smoothing_alpha', 0.2);
    parse(p, varargin{:});
    opts = p.Results;

    if isempty(opts.summary_csv) || ~isfile(opts.summary_csv)
        error('paper_export_module requires summary_csv (file not found).');
    end

    if ~exist(opts.results_dir, 'dir')
        mkdir(opts.results_dir);
    end

    out = struct();
    out.results_dir = opts.results_dir;

    if opts.export_olps
        export_olps_mat_results_all( ...
            'ipt_summary_csv', opts.summary_csv, ...
            'out_dir', fullfile(opts.results_dir, 'olps_mat'), ...
            'L_smoothing_alpha', opts.L_smoothing_alpha, ...
            'olps_dir', opts.olps_dir);
        out.export_olps = true;
    else
        out.export_olps = false;
    end
end
