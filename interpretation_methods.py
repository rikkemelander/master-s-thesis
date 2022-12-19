import numpy as np
import shap
import math


# Partial derivative influences
def pd_influence(diff, model, feat, data):
    var = np.sqrt(data.var(axis=0))[feat]
    diff_vec = diff(model, feat, data)
    return var*np.mean(diff_vec)


def pd_importance(diff, model, feat, data):
    """Calculate the partial derivative influence for differentiable regression models"""
    sd = data.var(axis=0)[feat]
    diff_vec = diff(model, feat, data)
    return np.sqrt(sd*np.mean(diff_vec**2))


def linear_reg_diff(model, feat, data):
    """Derivative of linear regression model"""
    beta = model.coef_
    return beta[feat]


def krr_reg_diff(model, feat, data):
    alpha = model.dual_coef_
    x_dist = np.repeat([model.X_fit_], len(data), axis=0)
    x_dist -= data.reshape(len(data), 1, 3)
    norm = np.linalg.norm(x_dist, axis=2)
    exp = np.exp(-norm**2/len(data[0, :]))
    inner = 2*x_dist[:, :, feat]/len(data[0, :])
    return np.sum(alpha*exp*inner, axis=1)


def non_diff(model, feat, data):
    new_data = []
    for col in range(len(data[0])):
        if col == feat:
            new_data.append(data[:, col])
        else:
            new_data.append(np.array([np.mean(data[:, col])] * len(data)))

    predicted_values = model.predict(np.array(new_data).T)

    result = []
    while len(data) > 1:
        denominator = data[0, feat] - data[1:, feat]
        numerator = (predicted_values[0] - predicted_values[1:])[denominator != 0]
        result += (numerator / denominator[denominator != 0]).tolist()
        predicted_values = predicted_values[1:len(data)]
        data = data[1:len(data)]

    return result


def true_linear_diff(model, feat, data):
    if feat == 0:
        return 1
    elif feat == 1:
        return -1
    else:
        return 0


def pd_tree_influence(tree, feat, data):
    """Calculate the partial derivative influence for single trees"""
    index = np.where(tree.feature == feat)[0]
    if len(index) == 0:
        return 0
    else:
        arr = tree.decision_path(np.float32(data)).toarray()[:, index].astype('float32').T
        arr[arr == 0.0] = np.nan
        sd = np.sqrt(np.nanvar(arr * data[:, feat], axis=1))
        dist = tree.value[tree.children_right[index]] - tree.value[tree.children_left[index]]
        return np.sum(sd*dist.reshape(len(dist)))/len(index)


def pd_rf_influence(model, feat, data):
    """Calculate the partial derivative influence for entire random forest model"""
    s = 0
    for i in model.estimators_:
        tree = i.tree_
        s += pd_tree_influence(tree, feat, data)
    return s / len(model.estimators_)


def pd_tree_influence_squared(tree, feat, data):
    """Calculate the partial derivative influence squared for single trees"""
    index = np.where(tree.feature == feat)[0]
    if len(index) == 0:
        return 0
    else:
        arr = tree.decision_path(np.float32(data)).toarray()[:, index].astype('float32').T
        arr[arr == 0.0] = np.nan
        var = np.nanvar(arr * data[:, feat], axis=1)
        dist = (tree.value[tree.children_right[index]] - tree.value[tree.children_left[index]])**2
        return np.sum(var*dist.reshape(len(dist)))/len(index)


def pd_rf_importance(model, feat, data):
    """Calculate the partial derivative influence for entire random forest model"""
    s = 0
    for i in model.estimators_:     # For every decision tree in random forest model
        tree = i.tree_      # Get the tree
        s += pd_tree_influence(tree, feat, data)    # Calculate the tree influence for the tree
    return np.sqrt(s / len(model.estimators_))


# Permutation feature importance
def perm_influence(predict, feat, data, n_permutations, max_predictions=100000):
    """Calculate permutation influence"""
    perm = np.repeat([data], n_permutations, axis=0)
    perm[:, :, feat] = np.apply_along_axis(np.random.permutation, 1, perm[:, :, feat])
    denominator = data[:, feat] - perm[:, :, feat]
    perm = perm[denominator != 0]
    if len(perm) > max_predictions:
        n_split = math.ceil(len(perm)/max_predictions)
        perm_predict = np.concatenate([predict(x) for x in np.array_split(perm, n_split)])
    else:
        perm_predict = predict(perm)
    x_predict = np.repeat([predict(data)], n_permutations, axis=0)[denominator != 0]
    numerator = (x_predict - perm_predict).ravel()
    return np.sum(numerator/denominator[denominator != 0])/(len(data)*n_permutations)


def perm_importance(predict, feat, data, n_permutations, max_predictions=100000):
    """Calculate permutation importance"""
    perm = np.repeat([data], n_permutations, axis=0)
    perm[:, :, feat] = np.apply_along_axis(np.random.permutation, 1, perm[:, :, feat])
    denominator = data[:, feat] - perm[:, :, feat]
    perm = perm[denominator != 0]
    if len(perm) > max_predictions:
        n_split = math.ceil(len(perm)/max_predictions)
        perm_predict = np.concatenate([predict(x) for x in np.array_split(perm, n_split)])
    else:
        perm_predict = predict(perm)
    x_predict = np.repeat([predict(data)], n_permutations, axis=0)[denominator != 0]
    numerator = (x_predict - perm_predict).ravel()
    return np.sqrt(np.sum((numerator/denominator[denominator != 0])**2)/(len(data)*n_permutations))


# SHAP values. As shap is very slow this one returns all three values.
def shap_influence(predict, feats, data, n_kmeans=100):
    explainer = shap.KernelExplainer(predict, shap.kmeans(data, n_kmeans))
    shap_values = explainer.shap_values(data)

    result = []
    for feat in feats:
        data_copy = data
        shap_values_copy = shap_values
        lst = []
        while len(data_copy) > 1:
            denominator = data_copy[0, feat] - data_copy[1:, feat]
            numerator = (shap_values_copy[0, feat] - shap_values_copy[1:, feat])[denominator != 0]
            lst += (numerator / denominator[denominator != 0]).tolist()
            shap_values_copy = shap_values_copy[1:len(data_copy)]
            data_copy = data_copy[1:len(data_copy)]
        result += [np.mean(lst)]
    return result


def shap_importance(predict, feats, data, n_kmeans=100):
    explainer = shap.KernelExplainer(predict, shap.kmeans(data, n_kmeans))
    shap_values = explainer.shap_values(data)

    result = []
    for feat in feats:
        data_copy = data
        shap_values_copy = shap_values
        lst = []
        while len(data_copy) > 1:
            denominator = data_copy[0, feat] - data_copy[1:, feat]
            numerator = (shap_values_copy[0, feat] - shap_values_copy[1:, feat])[denominator != 0]
            lst += (numerator / denominator[denominator != 0]).tolist()
            shap_values_copy = shap_values_copy[1:len(data_copy)]
            data_copy = data_copy[1:len(data_copy)]
        result += [np.sqrt(np.mean(np.array(lst)**2))]
    return result
