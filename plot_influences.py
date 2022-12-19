import read_csv as rc
import plot_functions as pf

#  EVALUATING THE INFLUENCE MEASURES


# Linear regression plots
def linear_plot_to_jpg():
    # Read data
    LI_influence = rc.read_influence_csv('data/linear_independent_influence.csv', linreg=True)
    LC_influence = rc.read_influence_csv('data/linear_correlated_influence.csv', linreg=True)
    NLI_influence = rc.read_influence_csv('data/non_linear_independent_influence.csv', linreg=True)
    NLC_influence = rc.read_influence_csv('data/non_linear_correlated_influence.csv', linreg=True)
    LNE_influence = rc.read_influence_csv('data/linear_no_extrapolation_influence.csv', linreg=True)
    LE_influence = rc.read_influence_csv('data/linear_extrapolation_influence.csv', linreg=False)
    NLNE_influence = rc.read_influence_csv('data/non_linear_no_extrapolation_influence.csv', linreg=True)
    NLE_influence = rc.read_influence_csv('data/non_linear_extrapolation_influence.csv', linreg=False)
    # Dependence comparison for linear data
    pf.linreg_grouped_box_plot(y_lst=[LI_influence['linear'], LC_influence['linear']],
                               plot_title='Linear regression on linear y',
                               legend=['x independent', 'x correlated'],
                               positions=[[1, 4, 7, 10], [2, 5, 8, 11]],
                               file_name='plots/linReg_dependence_influence.jpg')
    print('linReg_dependence_influence.jpg saved')

    # Extrapolation comparison for linear data
    pf.linreg_grouped_box_plot(y_lst=[LNE_influence['linear'], LE_influence['linear']],
                               plot_title='Linear regression on linear y',
                               legend=['x not extrapolated', 'x extrapolated'],
                               positions=[[1, 4, 7, 10], [2, 5, 8]],
                               file_name='plots/linReg_extrapolation_influence.jpg')
    print('linReg_extrapolation_influence.jpg saved')

    # Dependence comparison for non-linear data
    pf.linreg_grouped_box_plot(y_lst=[NLI_influence['linear'], NLC_influence['linear']],
                               plot_title='Linear regression on non-linear y',
                               legend=['x independent', 'x correlated'],
                               positions=[[1, 4, 7, 10], [2, 5, 8, 11]],
                               file_name='plots/non_linReg_dependence_influence.jpg')
    print('non_linReg_dependence_influence.jpg saved')

    # Extrapolation comparison for non-linear data
    pf.linreg_grouped_box_plot(y_lst=[NLNE_influence['linear'], NLE_influence['linear']],
                               plot_title='Linear regression on non-linear y',
                               legend=['x not extrapolated', 'x extrapolated'],
                               positions=[[1, 4, 7, 10], [2, 5, 8]],
                               file_name='plots/non_linReg_extrapolation_influence.jpg')
    print('non_linReg_extrapolation_influence.jpg saved')


# Kernel Ridge regression and Random Forest regression plots


def independence_plot_to_jpg():
    # Read influence data on real models
    true_model = rc.read_csv('data/true_model_influence.csv', ['pd', 'perm', 'shap'], 'feature', 'data_type')

    # Read Kernel Ridge and Random Forest independence data
    LI_influence = rc.read_influence_csv('data/linear_independent_influence.csv')
    NLI_influence = rc.read_influence_csv('data/non_linear_independent_influence.csv')

    # Linear independent comparison
    pf.grouped_box_plot(y_lst=[LI_influence['krr'], true_model['linear x_independent']],
                        plot_title='y linear, x independent',
                        legend=['Kernel Ridge regression', 'true model'],
                        fig_size=(5.0, 8.0),
                        file_name='plots/krr_independent_influence.jpg')
    print('krr_independent_influence.jpg saved')

    pf.grouped_box_plot(y_lst=[LI_influence['rf'], true_model['linear x_independent']],
                        plot_title='y linear, x independent',
                        legend=['Random Forest regression', 'true model'],
                        fig_size=(5.0, 8.0),
                        file_name='plots/rf_independent_influence.jpg')
    print('rf_independent_influence.jpg saved')

    # Non-linear independence comparison
    pf.grouped_box_plot(y_lst=[NLI_influence['krr'], true_model['non-linear x_independent']],
                        plot_title='y non-linear, x independent',
                        legend=['Kernel Ridge regression', 'true model'],
                        fig_size=(5.0, 8.0),
                        file_name='plots/non_krr_independent_influence.jpg')
    print('non_krr_independent_influence.jpg saved')

    pf.grouped_box_plot(y_lst=[NLI_influence['rf'], true_model['non-linear x_independent']],
                        plot_title='y non-linear, x independent',
                        legend=['Random Forest regression', 'true model'],
                        fig_size=(5.0, 8.0),
                        file_name='plots/non_rf_independent_influence.jpg')
    print('non_rf_independent_influence.jpg saved')


def correlation_plot_to_jpg():
    # Read influence data on real models
    true_model = rc.read_csv('data/true_model_influence.csv', ['pd', 'perm', 'shap'], 'feature', 'data_type')

    LC_influence = rc.read_influence_csv('data/linear_correlated_influence.csv')
    NLC_influence = rc.read_influence_csv('data/non_linear_correlated_influence.csv')

    # Linear correlation comparison
    pf.grouped_box_plot(y_lst=[LC_influence['krr'], true_model['linear x_correlated']],
                        plot_title='y linear, x correlated',
                        legend=['Kernel Ridge regression', 'true model'],
                        y_label='',
                        fig_size=(5.0, 8.0),
                        file_name='plots/krr_correlated_influence.jpg')
    print('krr_correlated_influence.jpg saved')

    pf.grouped_box_plot(y_lst=[LC_influence['rf'], true_model['linear x_correlated']],
                        plot_title='y linear, x correlated',
                        legend=['Random Forest regression', 'true model'],
                        y_label='',
                        fig_size=(5.0, 8.0),
                        file_name='plots/rf_correlated_influence.jpg')
    print('rf_correlated_influence.jpg saved')

    # Non-linear correlation comparison
    pf.grouped_box_plot(y_lst=[NLC_influence['krr'], true_model['non-linear x_correlated']],
                        plot_title='y non-linear, x correlated',
                        legend=['Kernel Ridge regression', 'true model'],
                        y_label='',
                        fig_size=(5.0, 8.0),
                        file_name='plots/non_krr_correlated_influence.jpg')
    print('non_krr_correlated_influence.jpg saved')

    pf.grouped_box_plot(y_lst=[NLC_influence['rf'], true_model['non-linear x_correlated']],
                        plot_title='y non-linear, x correlated',
                        legend=['Random Forest regression', 'true model'],
                        y_label='',
                        fig_size=(5.0, 8.0),
                        file_name='plots/non_rf_correlated_influence.jpg')
    print('non_rf_correlated_influence.jpg saved')


def extrapolation_plot_to_jpg():
    # Read influence data on real models
    true_model = rc.read_csv('data/true_model_influence.csv', ['pd', 'perm', 'shap'], 'feature', 'data_type')

    LNE_influence = rc.read_influence_csv('data/linear_no_extrapolation_influence.csv')
    LE_influence = rc.read_influence_csv('data/linear_extrapolation_influence.csv')
    NLNE_influence = rc.read_influence_csv('data/non_linear_no_extrapolation_influence.csv')
    NLE_influence = rc.read_influence_csv('data/non_linear_extrapolation_influence.csv')

    # Linear extrapolation comparison
    pf.grouped_box_plot(y_lst=[LNE_influence['krr'], LE_influence['krr'], true_model['linear x_no_extrapolation']],
                        plot_title='Kernel Ridge regression on linear y',
                        positions=[[1, 5, 9], [2, 6, 10], [3, 7, 11]],
                        legend=['non-extrapolated x', 'extrapolated x', 'true model'],
                        fig_size=(5.0, 8.0),
                        file_name='plots/krr_extrapolation_influence.jpg')
    print('krr_extrapolation_influence.jpg saved')

    pf.grouped_box_plot(y_lst=[LNE_influence['rf'], LE_influence['rf'], true_model['linear x_no_extrapolation']],
                        plot_title='Random Forest regression on linear y',
                        positions=[[1, 5, 9], [2, 6, 10], [3, 7, 11]],
                        legend=['non-extrapolated x', 'extrapolated x', 'true model'],
                        y_label='',
                        fig_size=(5.0, 8.0),
                        file_name='plots/rf_extrapolation_influence.jpg')
    print('rf_extrapolation_influence.jpg saved')

    # Non-linear no extrapolation comparison
    pf.grouped_box_plot(y_lst=[NLNE_influence['krr'], NLE_influence['krr'],
                               true_model['non-linear x_no_extrapolation']],
                        plot_title='Kernel Ridge regression on non-linear y',
                        positions=[[1, 5, 9], [2, 6, 10], [3, 7, 11]],
                        legend=['non-extrapolated x', 'extrapolated x', 'true model'],
                        fig_size=(5.0, 8.0),
                        file_name='plots/non_krr_extrapolation_influence.jpg')
    print('non_krr_extrapolation_influence.jpg saved')

    pf.grouped_box_plot(y_lst=[NLNE_influence['rf'], NLE_influence['rf'], true_model['non-linear x_no_extrapolation']],
                        plot_title='Random Forest regression on non-linear y',
                        positions=[[1, 5, 9], [2, 6, 10], [3, 7, 11]],
                        legend=['non-extrapolated x', 'extrapolated x', 'true model'],
                        y_label='',
                        fig_size=(5.0, 8.0),
                        file_name='plots/non_rf_extrapolation_influence.jpg')
    print('non_rf_extrapolation_influence.jpg saved')
