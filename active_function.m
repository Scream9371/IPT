function [w_YAR, Q_factor] = active_function(yar_weights_long, yar_weights_near, yar_ubah_long, yar_ubah_near, data, win_long, reverse_factor, risk_factor, q_value, L_long_history, L_near_history)
    % active_function - Three-state selection strategy for IPT model portfolio adjustment
    %
    %   [w_YAR, Q] = active_function(yar_weights_long, yar_weights_near, yar_ubah_long, yar_ubah_near, data, win_long)
    %   implements a three-state model selection strategy that adapts to varying
    %   market conditions by integrating recent trend, reversal potential and
    %   historical risk to calculate investment potential scores and model parameters.
    %
    %   This function is a core component of the Investment Potential Tracking (IPT)
    %   model, implementing the three-state selection strategy. It dynamically selects
    %   appropriate investment strategies based on overall market risk levels and outputs
    %   corresponding investment potential scores and model parameters Q_{t+1}.
    %
    % Inputs:
    %   yar_weights_long         - YAR factors from long-term window ((n - weight_inspect_wins) × m)
    %                              Yield-Adjusted Risk factors for long-term risk assessment \mathbf{w}_{t+1,\text{long-term}}
    %   yar_weights_near         - YAR factors from near-term window ((n - weight_inspect_wins / 2) × m)
    %                              Yield-Adjusted Risk factors for near-term risk assessment \mathbf{w}_{t+1,\text{near-term}}
    %   yar_ubah_long            - YAR under UBAH model from long-term window ((n - risk_inspect_wins) × 1)
    %                              Yield-Adjusted Risk of UBAH portfolio for long-term market analysis w_{t+1,\text{long-term}}^{\text{ubah}}
    %   yar_ubah_near            - YAR under UBAH model from near-term window ((n - risk_inspect_wins / 2) × 1)
    %                              Yield-Adjusted Risk of UBAH portfolio for near-term market analysis w_{t+1,\text{near-term}}^{\text{ubah}}
    %   data                     - Asset price data matrix (n × m)
    %                              The relative price for all assets over time periods \mathbf{x}_t
    %   win_long                 - Window size for long-term calculation (scalar)
    %                              Number of periods used for calculating long-term statistics d_l
    %   L_long_history            - Running percentile history for long-risk UBAH YAR
    %   L_near_history            - Running percentile history for near-risk UBAH YAR
    %
    % Outputs:
    %   w_YAR                    - Selected YAR weight matrix (n × m)
    %   Q                        - Model parameter Q_{t+1} vector (n × 1)
    %                              Represent different market states and strategies

    if nargin < 11 || isempty(L_near_history)
        L_near_history = L_long_history;
    end

    q = q_value;
    [datasets_T, datasets_N] = size(data);
    w_YAR = zeros(datasets_T, datasets_N);
    Q_factor = zeros(datasets_T, 1);

    for i = 1:datasets_T - win_long

        if isempty(L_long_history)
            L_long = 0;
        elseif i <= numel(L_long_history)
            L_long = L_long_history(i);
        else
            L_long = L_long_history(end);
        end

        if yar_ubah_long(i) <= q * L_long / 2
            Q_factor(i + win_long) = -2 * reverse_factor;
            w_YAR(i + win_long, :) = yar_weights_long(i, :);
        elseif yar_ubah_long(i) <= q * L_long
            Q_factor(i + win_long) = -reverse_factor;
            w_YAR(i + win_long, :) = yar_weights_long(i, :);
        else
            near_index = i + floor(win_long / 2);
            if near_index <= size(yar_ubah_near, 1)
                if isempty(L_near_history)
                    L_near = 0;
                elseif near_index <= numel(L_near_history)
                    L_near = L_near_history(near_index);
                else
                    L_near = L_near_history(end);
                end

                if yar_ubah_near(near_index) <= (1 - q) * L_near
                    Q_factor(i + win_long) = 0;
                    w_YAR(i + win_long, :) = yar_weights_near(near_index, :);
                elseif yar_ubah_near(near_index) <= (1 - q / 2) * L_near
                    Q_factor(i + win_long) = risk_factor;
                    w_YAR(i + win_long, :) = yar_weights_near(near_index, :);
                else
                    Q_factor(i + win_long) = 2 * risk_factor;
                    w_YAR(i + win_long, :) = yar_weights_near(near_index, :);
                end
            else
                % Default behavior if near_index is out of bounds
                Q_factor(i + win_long) = 0;
                w_YAR(i + win_long, :) = yar_weights_long(i, :);
            end
        end
    end
