import pandas as pd


def df_split(data_frame, value_labels, split_label):
    grouped = data_frame.groupby(split_label)
    dictionary = {}
    for group in data_frame[split_label].unique():
        subset = grouped.get_group(group)
        dictionary[group] = (subset.filter(items=value_labels))
    return dictionary


def df_to_array_split(dictionary, value_labels, split_label):
    if len(value_labels) == 1:
        for key, value in dictionary.items():
            value_arr = list(df_split(value, value_labels, split_label).values())
            dictionary[key] = list(map(lambda x: x.ravel(), map(lambda x: x.to_numpy(), value_arr)))
    else:
        for key, value in dictionary.items():
            value_arr = list(df_split(value, value_labels, split_label).values())
            dictionary[key] = list(map(lambda x: x.to_numpy(), value_arr))


def read_csv(csv_name, value_labels, split_label, df_split_label=None):
    df = pd.read_csv(csv_name)
    if df_split_label is None:
        df_lst = {'data': df}
        df_to_array_split(df_lst, value_labels, split_label)
    else:
        df_lst = df_split(df, value_labels + [split_label], df_split_label)
        df_to_array_split(df_lst, value_labels, split_label)
    return df_lst


def read_influence_csv(csv_name, linreg=False):
    value_labels = ['pd', 'perm', 'shap']

    if linreg is True:
        value_labels += ['linear_reg_value']

    df = read_csv(csv_name=csv_name,
                  value_labels=value_labels,
                  split_label='feature',
                  df_split_label='regression')
    return df
