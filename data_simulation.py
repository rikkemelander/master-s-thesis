import numpy as np
import pandas

data_x = {'feature_1': [],
          'feature_2': [],
          'feature_3': [],
          'data_type': [],
          'simulation': []}


def simulate_x():
    # Information for the simulation
    n_obs = 500  # number of samples in each simulation
    n_sim = 10

    zero_mean = np.array([0.0, 0.0, 0.0])

    cov_independent = np.array([[1, 0, 0],  # Covariance for independent simulation
                                [0, 1, 0],
                                [0, 0, 1]])

    cov_correlated = np.array([[1, 0.9, 0],  # Covariance for correlated simulation
                               [0.9, 1, 0],
                               [0, 0, 1]])
    for num in range(n_sim):
        x_extrapolation = np.random.uniform(0, 1, size=(n_obs, 3))
        dictionary = {'x_independent': np.random.multivariate_normal(zero_mean, cov_independent, n_obs),
                      'x_correlated': np.random.multivariate_normal(zero_mean, cov_correlated, n_obs),
                      'x_no_extrapolation': x_extrapolation[np.any([0.5 < x_extrapolation[:, 1],
                                                                    0.5 >= x_extrapolation[:, 0]], axis=0)],
                      'x_extrapolation': x_extrapolation}
        for key, value in dictionary.items():
            data_x['feature_1'] += list(value[:, 0])
            data_x['feature_2'] += list(value[:, 1])
            data_x['feature_3'] += list(value[:, 2])
            data_x['data_type'] += [key]*len(value)
            data_x['simulation'] += [num]*len(value)

    pandas.DataFrame(data_x).to_csv('data/x_data.csv', index=False)
    print('data/x_data.csv saved')


# Define the linear model
def linear(z):
    error = np.random.normal(0, 0.1, len(z))
    return z[:, 0] - z[:, 1] + error    # + 0*z[:, 2]


# Define help function for non-linear model
def help_function(z):
    if z[0] < 0.5:
        if z[1] < 0.5:
            return -1
        else:
            return 1
    else:
        if z[1] < 0.5:
            return 0
        else:
            return 2


# Define the non-linear model
def non_linear(z):
    error = np.random.normal(0, 0.1, len(z))
    return [help_function(x) for x in z] + error


def simulate_y():
    feature_values = np.stack([data_x['feature_1'], data_x['feature_2'], data_x['feature_3']], axis=1)
    data_y = {'y_linear': linear(feature_values),
              'y_non_linear': non_linear(feature_values),
              'x_type': data_x['data_type'],
              'simulation': data_x['simulation']}

    pandas.DataFrame(data_y).to_csv('data/y_data.csv', index=False)
    print('data/y_data.csv saved')
