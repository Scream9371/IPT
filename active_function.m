function [w_YAR, Q_factor] = active_function(yar_weights_long, yar_weights_near, yar_ubah_long, yar_ubah_near, data, win_long, reverse_factor, risk_factor)
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
    %   yar_weights_long         - YAR factors from long-term window (n × m)
    %                              Yield-Adjusted Risk factors for long-term risk assessment \mathbf{w}_{t+1,\text{long-term}}
    %   yar_weights_near         - YAR factors from near-term window (n × m)
    %                              Yield-Adjusted Risk factors for near-term risk assessment \mathbf{w}_{t+1,\text{near-term}}
    %   yar_ubah_long            - YAR under UBAH model from long-term window (n × 1)
    %                              Yield-Adjusted Risk of UBAH portfolio for long-term market analysis w_{t+1,\text{long-term}}^{\text{ubah}}
    %   yar_ubah_near            - YAR under UBAH model from near-term window (n × 1)
    %                              Yield-Adjusted Risk of UBAH portfolio for near-term market analysis w_{t+1,\text{near-term}}^{\text{ubah}}
    %   data                     - Asset price data matrix (n × m)
    %                              The relative price for all assets over time periods \mathbf{x}_t
    %   win_long                 - Window size for long-term calculation (scalar)
    %                              Number of periods used for calculating long-term statistics d_l
    %
    % Outputs:
    %   w_YAR                    - Selected YAR weight matrix (n × m)
    %   Q                        - Model parameter Q_{t+1} vector (n × 1)
    %                              Represent different market states and strategies

    L = 0.006; % Maximum YAR value under normal conditions
    q = 0.2; % Percentage values ​​to distinguish different market risk levels
    [datasets_T, datasets_N] = size(data);
    w_YAR = zeros(datasets_T, datasets_N);
    Q_factor = zeros(datasets_T, 1);

    for i = 1:datasets_T - win_long

        if yar_ubah_long(i) <= q * L / 2
            Q_factor(i + win_long) = -2 * reverse_factor;
            w_YAR(i + win_long, :) = yar_weights_long(i, :);
        elseif yar_ubah_long(i) <= q * L
            Q_factor(i + win_long) = -reverse_factor;
            w_YAR(i + win_long, :) = yar_weights_long(i, :);
        else

            if yar_ubah_near(i + win_long / 2) <= (1 - q) * L
                Q_factor(i + win_long) = 0;
                w_YAR(i + win_long, :) = yar_weights_near(i + win_long / 2, :);
            elseif yar_ubah_near(i + win_long / 2) <= (1 - q / 2) * L
                Q_factor(i + win_long) = risk_factor;
                w_YAR(i + win_long, :) = yar_weights_near(i + win_long / 2, :);
            else
                Q_factor(i + win_long) = 2 * risk_factor;
                w_YAR(i + win_long, :) = yar_weights_near(i + win_long / 2, :);
            end

        end

    end
