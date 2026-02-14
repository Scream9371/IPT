function out_dir = run(varargin)
%run One command entry for IPT paper mainline.

    repo_root = fileparts(mfilename('fullpath'));
    addpath(fullfile(repo_root, 'ipt_mainline'));
    out_dir = run_core(varargin{:});
end
