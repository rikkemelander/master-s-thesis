import read_csv as rc
import influence_calculation_functions as icf


def get_data():
    x_data = rc.read_csv('data/x_data.csv', ['feature_1', 'feature_2', 'feature_3'], 'simulation', 'data_type')
    y_linear_data = rc.read_csv('data/y_data.csv', ['y_linear'], 'simulation', 'x_type')
    y_non_linear_data = rc.read_csv('data/y_data.csv', ['y_non_linear'], 'simulation', 'x_type')
    return x_data, y_linear_data, y_non_linear_data


def pd_rf_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.rf_pd_test(x_dict=x_data, y_dict=y_linear_data, n_tests=5, feats=[0, 1, 2], csv_name='data/rf_pd_test.csv')


def pd_compare_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    icf.pd_compare_test(x_lst=x_data['x_independent'][:5], y_lst=y_linear_data['x_independent'][:5], feats=[0, 1, 2],
                        csv_name='data/pd_compare_test.csv')


def perm_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    permutation_lst = list(range(1, 101, 1))
    icf.permutation_test(x_dict=x_data, y_dict={'linear': y_linear_data, 'non_linear': y_non_linear_data}, feat=1,
                         permutation_lst=permutation_lst, csv_name='data/perm_test.csv')


def perm_rf_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    rf_permutation_lst = list(range(1, 2002, 20))
    icf.rf_permutation_test(x_data=x_data['x_no_extrapolation'][0], y_data=y_non_linear_data['x_no_extrapolation'][0],
                            feat=1,
                            permutation_lst=rf_permutation_lst, csv_name='data/rf_perm_test.csv')


def shap_to_csv():
    x_data, y_linear_data, y_non_linear_data = get_data()
    kmeans_lst = list(range(10, 201, 10))
    icf.shap_kmeans_test(x_dict=x_data, y_dict={'linear': y_linear_data, 'non_linear': y_non_linear_data},
                         feats=[0, 1, 2],
                         feat=1, kmeans_lst=kmeans_lst, csv_name='data/shap_test.csv')
