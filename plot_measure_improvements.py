import read_csv as rc
import plot_functions as pf
import pandas

# TESTING PARTS OF INFLUENCE MEASURES


def pd_rf_plot_to_jpg():
    # Read data for test plots
    df_rf_pd = rc.read_csv(csv_name='data/rf_pd_test.csv',
                           value_labels=['x_independent', 'x_correlated', 'x_no_extrapolation', 'x_extrapolation'],
                           split_label='feature')
    # Testing original Partial Derivative influence for Random forest
    pf.rf_pd_plot(y_lst=[[df_rf_pd['data'][0]], [df_rf_pd['data'][1]], [df_rf_pd['data'][2]]],
                  plot_title='Pd measure test for Random Forest regression',
                  file_name='plots/rf_pd_test.jpg')
    print('rf_pd_test.jpg saved')


def pd_compare_plot_to_jpg():
    df_pd_compare = rc.read_csv(csv_name='data/pd_compare_test.csv',
                                value_labels=['linear', 'krr', 'rf'],
                                split_label='feature',
                                df_split_label='pd_measure')
    # Comparison of original and new Partial Derivative influence
    pf.grouped_box_plot(y_lst=[df_pd_compare['original_pd'], df_pd_compare['new_pd']],
                        plot_title='Comparison of Partial Derivative measures',
                        legend=['Original pd', 'New pd'],
                        x_labels=['Linear', 'Krr', 'rf'],
                        file_name='plots/pd_compare_test.jpg')
    print('pd_compare_test.jpg saved')


def perm_plot_to_jpg():
    df_perm = rc.read_csv(csv_name='data/perm_test.csv',
                          value_labels=['linear', 'krr', 'rf'],
                          split_label='y_type',
                          df_split_label='x_type')
    # Testing number of permutations in Permutation influence
    permutation_lst = list(range(1, 101, 1))
    pf.permutation_plot(x_lst=permutation_lst, y_lst=list(df_perm.values()),
                        plot_title='Testing number of permutations in Permutation measures',
                        file_name='plots/permutation_test.jpg')
    print('permutation_test.jpg saved')


def perm_rf_plot_to_jpg():
    df_rf_perm = pandas.read_csv('data/rf_perm_test.csv')
    # Testing number of permutations in Permutation influence for specifically Random Forest
    pf.rf_permutation_plot(x_lst=df_rf_perm['n_permutations'], y_lst=df_rf_perm['rf'],
                           plot_title='Number of permutations for Random Forest regression',
                           file_name='plots/rf_permutation_test.jpg')
    print('rf_permutation_test.jpg saved')


def shap_plot_to_jpg():
    df_shap = rc.read_csv(csv_name='data/shap_test.csv',
                          value_labels=['linear', 'krr', 'rf'],
                          split_label='y_type',
                          df_split_label='x_type')
    # Testing number of k-means in SHAP influence
    k_means_lst = list(range(10, 201, 10))
    pf.shap_plot(x_lst=k_means_lst, y_lst=list(df_shap.values()),
                 plot_title='Testing number of k-means in SHAP value measures',
                 file_name='plots/shap_test.jpg')
    print('shap_test.jpg saved')
