function name = ipt_state_name(code)
% ipt_state_name - Map numeric state code to a string label.

    if ~isfinite(code)
        name = "warmup";
        return;
    end
    switch round(code)
        case 1
            name = "reversal_strong";
        case 2
            name = "reversal";
        case 3
            name = "normal";
        case 4
            name = "risk";
        case 5
            name = "risk_strong";
        otherwise
            name = "unknown";
    end
end

