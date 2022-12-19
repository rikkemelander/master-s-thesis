import read_csv as rc
import influence_calculation_functions as icf


def get_data():
    x_data = rc.read_csv('data/x_data.csv', ['feature_1', 'feature_2', 'feature_3'], 'simulation', 'data_type')
    y_linear_data = rc.read_csv('data/y_data.csv', ['y_linear'], 'simulation', 'x_type')
    y_non_linear_data = rc.read_csv('data/y_data.csv', ['y_non_linear'], 'simulation', 'x_type')
    return x_data, y_linear_data, y_non_linear_data


# Linear
def linear_independent_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.influence_normal_data(x_data=x_data['x_independent'], y_data=y_linear_data['x_independent'], feats=[0, 1, 2],
                              csv_name='data/linear_independent_influence.csv')


def linear_correlated_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.influence_normal_data(x_data=x_data['x_correlated'], y_data=y_linear_data['x_correlated'], feats=[0, 1, 2],
                              csv_name='data/linear_correlated_influence.csv')


def linear_extrapolation_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.influence_extrapolation_data(x_no_extrapolation=x_data['x_no_extrapolation'],
                                     y_no_extrapolation=y_linear_data['x_no_extrapolation'],
                                     x_extrapolation=x_data['x_extrapolation'], feats=[0, 1, 2],
                                     no_extrapolation_csv_name='data/linear_no_extrapolation_influence.csv',
                                     extrapolation_csv_name='data/linear_extrapolation_influence.csv')


# Non-linear
def non_linear_independent_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.influence_normal_data(x_data=x_data['x_independent'], y_data=y_non_linear_data['x_independent'],
                              feats=[0, 1, 2], csv_name='data/non_linear_independent_influence.csv')


def non_linear_correlated_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.influence_normal_data(x_data=x_data['x_correlated'], y_data=y_non_linear_data['x_correlated'],
                              feats=[0, 1, 2], csv_name='data/non_linear_correlated_influence.csv')


def non_linear_extrapolation_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.influence_extrapolation_data(x_no_extrapolation=x_data['x_no_extrapolation'],
                                     y_no_extrapolation=y_non_linear_data['x_no_extrapolation'],
                                     x_extrapolation=x_data['x_extrapolation'], feats=[0, 1, 2],
                                     no_extrapolation_csv_name='data/non_linear_no_extrapolation_influence.csv',
                                     extrapolation_csv_name='data/non_linear_extrapolation_influence.csv')


def true_model_influence_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.influence_true_data(x_dict=x_data, feats=[0, 1, 2], n_simulations=10, csv_name='data/true_model_influence.csv')
