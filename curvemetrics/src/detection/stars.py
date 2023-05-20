import numpy as np
from scipy.stats import t

def stars(data, col, l, prob=0.95, merge=False):
    vals = data[col].values
    t_crit = t.ppf(prob, 2*l - 2)
  
    # Assuming we have a similar rust_rodionov function in Python
    RSI = rodionov(vals, t_crit, l)
    RSI = np.append(RSI, np.zeros(l - 1))
  
    # Creates results dataframe
    output = data.copy()
    output['RSI'] = RSI
    if not merge:
        output = output[output['RSI'] > 0]
    return output

def calculate_rsi(regime, shift_boundary, is_down, l, var_l):
    rsi = 0.
    for val in regime:
        x_i_star = shift_boundary - val if is_down else val - shift_boundary
        rsi += x_i_star / (l * np.sqrt(var_l))
        if rsi < 0.:
            rsi = 0.
            break
    return rsi

def rodionov(vals, t_crit, l):
    results = []

    var_l = 0.
    for i in range(len(vals) - l):
        sum_l = np.var(vals[i:i+l])
        var_l += sum_l / l
    var_l /= (len(vals) - l)

    diff = t_crit * np.sqrt((2. * var_l) / l)

    regime_length = l
    regime_mean = np.mean(vals[:l])
    boundary_upper = regime_mean + diff
    boundary_lower = regime_mean - diff

    cand_len = len(vals) - l + 1
    candidates = vals[:cand_len]

    for i, val in enumerate(candidates):

        if val < boundary_lower:
            rsi = calculate_rsi(vals[i:i+l], boundary_lower, True, l, var_l)
        elif val > boundary_upper:
            rsi = calculate_rsi(vals[i:i+l], boundary_upper, False, l, var_l)
        else:
            rsi = 0.

        if rsi > 0.:  # regime boundary found; start new regime
            results.append(rsi)
            regime_length = l
            regime_mean = np.mean(vals[i:i+l])
        else:  # regime test failed; add value to current regime
            results.append(0.)
            regime_length += 1
            regime_mean = (regime_mean * regime_length + val) / regime_length
        boundary_lower = regime_mean - diff
        boundary_upper = regime_mean + diff

    return results
