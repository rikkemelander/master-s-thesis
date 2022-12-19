import interpretation_methods as im
import pandas
import numpy as np
import data_simulation as ds
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


def fit_models(x_data, y_data):
    # fit linear model
    linear = LinearRegression()
    linear.fit(x_data, y_data)

    # fit krr model
    krr = KernelRidge(alpha=1.0, kernel='rbf')
    krr.fit(x_data, y_data)

    # fit rf model
    rf = RandomForestRegressor()
    rf.fit(x_data, y_data)

    return linear, krr, rf


def rf_pd_test(x_dict, y_dict, n_tests, feats, csv_name):
    # create dictionary for csv file
    dictionary = {'x_independent': [],
                  'x_correlated': [],
                  'x_no_extrapolation': [],
                  'x_extrapolation': [],
                  'feature': []}
    for test in range(n_tests):
        for key in list(x_dict.keys()):
            if key == 'x_extrapolation':
                rf = RandomForestRegressor()
                rf.fit(x_dict['x_no_extrapolation'][test], y_dict['x_no_extrapolation'][test])
            else:
                rf = RandomForestRegressor()
                rf.fit(x_dict[key][test], y_dict[key][test])
            dictionary[key] += list(map(lambda x: im.pd_rf_influence(model=rf, feat=x, data=x_dict[key][test]), feats))

        print('simulation {} out of {} done'.format(test + 1, n_tests))

    dictionary['feature'] = feats*n_tests

    # save dictionary as csv
    pandas.DataFrame(dictionary).to_csv(csv_name, index=False)
    print('{} saved'.format(csv_name))


def pd_compare_test(x_lst, y_lst, feats, csv_name):
    # create dictionary for csv file
    dictionary = {'linear': [],
                  'krr': [],
                  'rf': [],
                  'pd_measure': [],
                  'feature': []}
    for index in range(len(x_lst)):
        linear, krr, rf = fit_models(x_lst[index], y_lst[index])

        dictionary['linear'] += list(map(lambda x: im.pd_influence(diff=im.linear_reg_diff, model=linear, feat=x,
                                                                   data=x_lst[index]), feats))
        dictionary['krr'] += list(map(lambda x: im.pd_influence(diff=im.krr_reg_diff, model=krr, feat=x,
                                                                data=x_lst[index]), feats))
        dictionary['rf'] += list(map(lambda x: im.pd_rf_influence(model=rf, feat=x, data=x_lst[index]), feats))

        dictionary['linear'] += list(map(lambda x: im.pd_influence(diff=im.non_diff, model=linear, feat=x,
                                                                   data=x_lst[index]), feats))
        dictionary['krr'] += list(map(lambda x: im.pd_influence(diff=im.non_diff, model=krr, feat=x,
                                                                data=x_lst[index]), feats))
        dictionary['rf'] += list(map(lambda x: im.pd_influence(diff=im.non_diff, model=rf, feat=x,
                                                               data=x_lst[index]), feats))

        dictionary['pd_measure'] += ['original_pd']*3 + ['new_pd']*3
        dictionary['feature'] += feats*2

        print('simulation {} out of {} saved'.format(index + 1, len(x_lst)))

    # save dictionary as csv
    pandas.DataFrame(dictionary).to_csv(csv_name, index=False)
    print('{} saved'.format(csv_name))


def permutation_test(x_dict, y_dict, feat, permutation_lst, csv_name):
    dictionary = {'linear': [],
                  'krr': [],
                  'rf': [],
                  'x_type': [],
                  'y_type': []}
    for y_key in list(y_dict.keys()):
        for x_key in list(x_dict.keys()):
            if x_key == 'x_extrapolation':
                linear, krr, rf = fit_models(x_dict['x_no_extrapolation'][0], y_dict[y_key]['x_no_extrapolation'][0])
            else:
                linear, krr, rf = fit_models(x_dict[x_key][0], y_dict[y_key][x_key][0])

            # calculate perm influences for linear, krr and rf for all elements in permutation_lst
            dictionary['linear'] += list(map(lambda x: im.perm_influence(predict=linear.predict, feat=feat,
                                                                         data=x_dict[x_key][0], n_permutations=x),
                                             permutation_lst))
            dictionary['krr'] += list(map(lambda x: im.perm_influence(predict=krr.predict, feat=feat,
                                                                      data=x_dict[x_key][0], n_permutations=x),
                                          permutation_lst))
            dictionary['rf'] += list(map(lambda x: im.perm_influence(predict=rf.predict, feat=feat,
                                                                     data=x_dict[x_key][0], n_permutations=x),
                                         permutation_lst))

            dictionary['x_type'] += [x_key] * len(permutation_lst)
            dictionary['y_type'] += [y_key] * len(permutation_lst)

            print('{} {} done'.format(y_key, x_key))

    # save dictionary as csv
    pandas.DataFrame(dictionary).to_csv(csv_name, index=False)
    print('{} saved'.format(csv_name))


def rf_permutation_test(x_data, y_data, feat, permutation_lst, csv_name):
    dictionary = {'rf': [],
                  'n_permutations': []}

    # fit rf model
    rf = RandomForestRegressor()
    rf.fit(x_data, y_data)

    # calculate perm influences rf for all elements in permutation_lst
    dictionary['rf'] += list(map(lambda x: im.perm_influence(predict=rf.predict, feat=feat, data=x_data,
                                                             n_permutations=x), permutation_lst))
    dictionary['n_permutations'] += permutation_lst

    # save dictionary as csv
    pandas.DataFrame(dictionary).to_csv(csv_name, index=False)
    print('{} saved'.format(csv_name))


def shap_kmeans_test(x_dict, y_dict, feats, feat, kmeans_lst, csv_name):
    dictionary = {'linear': [],
                  'krr': [],
                  'rf': [],
                  'x_type': [],
                  'y_type': []}
    for y_key in list(y_dict.keys()):
        for x_key in list(x_dict.keys()):
            if x_key == 'x_extrapolation':
                linear, krr, rf = fit_models(x_dict['x_no_extrapolation'][0], y_dict[y_key]['x_no_extrapolation'][0])
            else:
                linear, krr, rf = fit_models(x_dict[x_key][0], y_dict[y_key][x_key][0])

            # calculate perm influences for linear, krr and rf for all elements in permutation_lst
            dictionary['linear'] += list(map(lambda x: im.shap_influence(predict=linear.predict, feats=feats,
                                                                         data=x_dict[x_key][0], n_kmeans=x)[feat],
                                             kmeans_lst + [len(x_dict[x_key][0])]))

            dictionary['krr'] += list(map(lambda x: im.shap_influence(predict=krr.predict, feats=feats,
                                                                      data=x_dict[x_key][0], n_kmeans=x)[feat],
                                          kmeans_lst + [len(x_dict[x_key][0])]))

            dictionary['rf'] += list(map(lambda x: im.shap_influence(predict=rf.predict, feats=feats,
                                                                     data=x_dict[x_key][0], n_kmeans=x)[feat],
                                         kmeans_lst + [len(x_dict[x_key][0])]))

            dictionary['x_type'] += [x_key] * (len(kmeans_lst)+1)
            dictionary['y_type'] += [y_key] * (len(kmeans_lst)+1)

            print('{} {} done'.format(y_key, x_key))

    # save dictionary as csv
    pandas.DataFrame(dictionary).to_csv(csv_name, index=False)
    print('{} saved'.format(csv_name))


def influence_help_function(dictionary, linear, krr, rf, feats, x):
    linear_shap = im.shap_influence(linear.predict, feats, x, 1)
    krr_shap = im.shap_influence(krr.predict, feats, x, 100)
    rf_shap = im.shap_influence(rf.predict, feats, x, 150)
    for index in range(len(feats)):
        dictionary['pd'].append(im.pd_influence(im.linear_reg_diff, linear, feats[index], x))
        dictionary['perm'].append(im.perm_influence(linear.predict, feats[index], x, 50))
        dictionary['shap'].append(linear_shap[index])
        dictionary['linear_reg_value'].append(linear.coef_[index])
        dictionary['regression'].append('linear')

        dictionary['pd'].append(im.pd_influence(im.krr_reg_diff, krr, feats[index], x))
        dictionary['perm'].append(im.perm_influence(krr.predict, feats[index], x, 50))
        dictionary['shap'].append(krr_shap[index])
        dictionary['linear_reg_value'].append(None)
        dictionary['regression'].append('krr')

        dictionary['pd'].append(im.pd_influence(im.non_diff, rf, feats[index], x))
        dictionary['perm'].append(im.perm_influence(rf.predict, feats[index], x, 1500))
        dictionary['shap'].append(rf_shap[index])
        dictionary['linear_reg_value'].append(None)
        dictionary['regression'].append('rf')

        dictionary['feature'] += [feats[index]] * 3


def influence_normal_data(x_data, y_data, feats, csv_name):
    dictionary = {'pd': [],
                  'perm': [],
                  'shap': [],
                  'linear_reg_value': [],
                  'regression': [],
                  'feature': []}

    for index in range(len(x_data)):
        linear, krr, rf = fit_models(x_data[index], y_data[index])
        influence_help_function(dictionary=dictionary, linear=linear, krr=krr, rf=rf, feats=feats, x=x_data[index])
        print('simulation {} out of {} done'.format(index+1, len(x_data)))

    pandas.DataFrame(dictionary).to_csv(csv_name, index=False)
    print('{} done'.format(csv_name))


def influence_extrapolation_data(x_no_extrapolation, y_no_extrapolation, x_extrapolation, feats,
                                 no_extrapolation_csv_name, extrapolation_csv_name):
    dictionary_no_extrapolation = {'pd': [],
                                   'perm': [],
                                   'shap': [],
                                   'linear_reg_value': [],
                                   'regression': [],
                                   'feature': []}
    dictionary_extrapolation = {'pd': [],
                                'perm': [],
                                'shap': [],
                                'linear_reg_value': [],
                                'regression': [],
                                'feature': []}

    for index in range(len(x_no_extrapolation)):
        linear, krr, rf = fit_models(x_no_extrapolation[index], y_no_extrapolation[index])

        influence_help_function(dictionary=dictionary_no_extrapolation, linear=linear, krr=krr, rf=rf, feats=feats,
                                x=x_no_extrapolation[index])
        influence_help_function(dictionary=dictionary_extrapolation, linear=linear, krr=krr, rf=rf, feats=feats,
                                x=x_extrapolation[index])
        print('simulation {} out of {} done'.format(index + 1, len(x_no_extrapolation)))

    pandas.DataFrame(dictionary_no_extrapolation).to_csv(no_extrapolation_csv_name, index=False)
    print('{} saved'.format(no_extrapolation_csv_name))

    pandas.DataFrame(dictionary_extrapolation).to_csv(extrapolation_csv_name, index=False)
    print('{} saved'.format(extrapolation_csv_name))


def true_linear(x):
    if len(x[0]) == 1:
        return x[0] - x[1]
    else:
        return x[:, 0] - x[:, 1]


def true_non_linear(x):
    return np.array([ds.help_function(z) for z in x])


def true_non_linear_diff(model, feat, data):
    new_data = []
    for col in range(len(data[0])):
        if col == feat:
            new_data.append(data[:, col])
        else:
            new_data.append(np.array([np.mean(data[:, col])] * len(data)))

    predicted_values = model(np.array(new_data).T)

    result = []
    while len(data) > 1:
        denominator = data[0, feat] - data[1:, feat]
        numerator = (predicted_values[0] - predicted_values[1:])[denominator != 0]
        result += (numerator / denominator[denominator != 0]).tolist()
        predicted_values = predicted_values[1:len(data)]
        data = data[1:len(data)]

    return result


def influence_true_data(x_dict, feats, n_simulations, csv_name):
    dictionary = {'pd': [],
                  'perm': [],
                  'shap': [],
                  'feature': [],
                  'data_type': []}
    for x_key in list(x_dict.keys()):
        for index in range(n_simulations):
            dictionary['pd'] += list(map(lambda x: im.pd_influence(diff=im.true_linear_diff, model=true_linear,
                                                                   feat=x, data=x_dict[x_key][index]), feats))
            dictionary['perm'] += list(map(lambda x: im.perm_influence(predict=true_linear, feat=x, n_permutations=50,
                                                                       data=x_dict[x_key][index]), feats))
            dictionary['shap'] += im.shap_influence(predict=true_linear, feats=feats, data=x_dict[x_key][index],
                                                    n_kmeans=100)
            dictionary['feature'] += feats
            dictionary['data_type'] += ['linear {}'.format(x_key)]*len(feats)

            dictionary['pd'] += list(map(lambda x: im.pd_influence(diff=true_non_linear_diff, model=true_non_linear,
                                                                   feat=x, data=x_dict[x_key][index]), feats))
            dictionary['perm'] += list(map(lambda x: im.perm_influence(predict=true_non_linear, n_permutations=50,
                                                                       feat=x, data=x_dict[x_key][index]), feats))
            dictionary['shap'] += im.shap_influence(predict=true_non_linear, feats=feats, data=x_dict[x_key][index],
                                                    n_kmeans=100)
            dictionary['feature'] += feats
            dictionary['data_type'] += ['non-linear {}'.format(x_key)]*len(feats)
            print('simulation {} out of {} done'.format(index + 1, n_simulations))
        print('{} saved'.format(x_key))

    pandas.DataFrame(dictionary).to_csv(csv_name, index=False)
    print('{} saved'.format(csv_name))
