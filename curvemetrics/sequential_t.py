import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def calculate_slope(series):
    # Fit a simple linear regression
    X = sm.add_constant(np.arange(len(series)))
    model = sm.OLS(series, X)
    results = model.fit()
    slope = results.params[1]
    return slope

def sequential_t_test(df, window_size):
    # Calculate the slope for each rolling window
    df['slope'] = df['returns'].rolling(window_size).apply(calculate_slope, raw=False)

    # Initialize a list to store p-values
    p_values = [np.nan] * window_size

    # Perform the sequential t-test
    for i in range(window_size, len(df)):
        # Calculate the t-statistic and two-tailed p-value
        t_stat, p_value = stats.ttest_ind(df['slope'].iloc[i-window_size:i], 
                                           df['slope'].iloc[i-window_size+1:i+1], 
                                           equal_var=False)
        p_values.append(p_value)

    # Add the p-values to the dataframe
    df['p_value'] = p_values

    return df

# Assume df is a DataFrame with a 'returns' column
window_size = 60  # for example
df = sequential_t_test(df, window_size)
