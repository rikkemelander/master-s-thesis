import matplotlib.pyplot as plt
import numpy as np


def get_default_colors():
    default_colors = {'red': [0.565, 0.12, 0.118],
                      'green': [0.2, 0.37, 0.21],
                      'grey': [0.3, 0.3, 0.3]}
    return default_colors


def ax_settings(ax_lst, n_row=1, n_col=1, sub_titles=None, grid_axis='both'):
    for index in range(len(ax_lst)):
        ax_lst[index].tick_params(labelsize=9, bottom=False, left=False, pad=1)
        ax_lst[index].grid(axis=grid_axis, linestyle=':')
        ax_lst[index].spines.right.set_visible(False)
        ax_lst[index].spines.top.set_visible(False)
        if sub_titles is not None:
            ax_lst[index].set_title(sub_titles[index], fontsize=10, pad=10, loc='left', fontname='serif')
        if n_row > 1:
            if index not in np.array(range(len(ax_lst)))[-n_col:]:
                ax_lst[index].tick_params(labelbottom=False)


def rf_pd_plot(y_lst, plot_title, file_name):

    # Create figure
    fig, ax = plt.subplots()

    # Plot title and labels
    fig.suptitle(plot_title, x=0.55, y=0.99, fontsize=12, ha='center', fontname='serif')
    ax.set_ylabel('influence', fontsize=9, fontname='serif')

    # Layout settings
    ax_settings([ax], grid_axis='y')

    # For legends
    dictionary = {}

    # Position of box plots
    positions = [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]

    # Create box plots for each dataset in lst_data
    for group in range(len(y_lst)):
        color = list(get_default_colors().values())[group]
        transparent_color = list(get_default_colors().values())[group] + [0.8]
        dictionary['{}'.format(group)] = ax.boxplot(y_lst[group][0], positions=positions[group], patch_artist=True,
                                                    widths=0.7, boxprops=dict(facecolor=transparent_color, color=color),
                                                    capprops=dict(color=color), whiskerprops=dict(color=color),
                                                    flierprops=dict(color=color, markeredgecolor=color),
                                                    medianprops=dict(color=color))

    # Create legend in top of figure
    fig.legend(list(map(lambda x: x["boxes"][0], list(dictionary.values()))), ['Feature 1', 'Feature 2', 'Feature 3'],
               prop={'size': 9}, loc='upper center', bbox_to_anchor=[0.55, 0.95], ncol=3)
    ax.set_xticks([2, 6, 10, 14], ['x independent', 'x correlated', 'x not extrapolated', 'x extrapolated'],
                  fontsize=9, fontname='serif', ha='center')
    [ax.axvline(x, color='black', linestyle='--', linewidth='1') for x in [4, 8, 12]]

    # Make layout tight
    plt.tight_layout(pad=1.5)

    # Save plot as file_name
    plt.savefig(file_name)
    plt.close()


def permutation_plot(x_lst, y_lst, plot_title, file_name):

    # Create figure
    fig, ax = plt.subplots(len(y_lst), len(y_lst[0]), figsize=(8.0, 8.0))

    # Plot title and labels
    fig.suptitle(plot_title, x=0.55, y=0.99, fontsize=13, ha='center', fontname='serif')
    fig.text(0.54, 0.01, 'permutations', fontsize=9, ha='center', fontname='serif')
    fig.text(0.008, 0.48, 'influence', fontsize=9, va='center', rotation='vertical', fontname='serif')

    # Layout settings
    ax_settings(ax.flatten(), n_row=len(y_lst), n_col=len(y_lst[0]),
                sub_titles=['y linear, x independent', 'y non-linear, x independent',
                            'y linear, x correlated', 'y non-linear, x correlated',
                            'y linear, x not extrapolated', 'y non-linear, x not extrapolated',
                            'y linear, x extrapolated', 'y non-linear, x extrapolated'])

    # For legends
    dictionary = {}

    # Specify colors
    colors = [get_default_colors()['grey'], get_default_colors()['green'], get_default_colors()['red']]

    # Create line plot for each dataset
    for row in range(len(y_lst)):
        for col in range(len(y_lst[0])):
            for index in range(len(y_lst[row][col][0])):
                dictionary['{}'.format(index)], = ax[row, col].plot(x_lst, np.array(y_lst[row][col])[:, index],
                                                                    linestyle=':', marker='o', markersize=2,
                                                                    color=colors[index], linewidth=1)

    # Create legend in top of figure
    fig.legend(list(dictionary.values()), ['linear regression', 'krr regression', 'rf regression'],
               bbox_to_anchor=[0.55, 0.91], loc='lower center', prop={'size': 9}, ncol=3)

    # Save figure
    plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=1.5)
    plt.savefig(file_name)
    plt.close()


def rf_permutation_plot(x_lst, y_lst, plot_title, file_name, x_label='permutations', y_label='influence'):

    # Create plot
    fig, ax = plt.subplots()

    # Plot title and labels
    fig.suptitle(plot_title, x=0.55, y=0.99, fontsize=12, ha='center', fontname='serif')
    ax.set_ylabel(y_label, fontsize=10, fontname='serif')
    ax.set_xlabel(x_label, fontsize=10, fontname='serif')

    # Layout settings
    ax_settings([ax])

    # Add line plot
    ax.plot(x_lst, y_lst, color='#981C1F', linestyle=':', marker='o', markersize=3)

    # Save figure
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def shap_plot(x_lst, y_lst, plot_title, file_name):

    # Create figure
    fig, ax = plt.subplots(len(y_lst), len(y_lst[0]), figsize=(8.0, 8.0))

    # Plot title and labels
    fig.suptitle(plot_title, x=0.55, y=0.99, fontsize=13, ha='center', fontname='serif')
    fig.text(0.54, 0.01, 'number of k-means', fontsize=9, ha='center', fontname='serif')
    fig.text(0.008, 0.48, 'influence', fontsize=9, va='center', rotation='vertical', fontname='serif')

    # Layout settings
    ax_settings(ax.flatten(), n_row=len(y_lst), n_col=len(y_lst[0]),
                sub_titles=['y linear, x independent', 'y non-linear, x independent',
                            'y linear, x correlated', 'y non-linear, x correlated',
                            'y linear, x not extrapolated', 'y non-linear, x not extrapolated',
                            'y linear, x extrapolated', 'y non-linear, x extrapolated'])

    # For labels
    dictionary = {}

    # Specify colors
    colors = [get_default_colors()['grey'], get_default_colors()['green'], get_default_colors()['red']]

    # Create line plot for each dataset
    for row in range(len(y_lst)):
        for col in range(len(y_lst[0])):
            for index in range(len(y_lst[row][col][0])):
                y = np.array(y_lst[row][col])
                dictionary['{}'.format(index)], = ax[row, col].plot(x_lst, y[:-1, index], linestyle=':', marker='o',
                                                                    markersize=2, color=colors[index], linewidth=1)
                ax[row, col].plot([x_lst[0], x_lst[-1]], [y[-1, index], y[-1, index]], linestyle='-',
                                  color=colors[index], linewidth=1)

    # Create legend in top of figure
    fig.legend(list(dictionary.values()), ['linear regression', 'krr regression', 'rf regression'],
               bbox_to_anchor=[0.55, 0.91], loc='lower center', prop={'size': 9}, ncol=3)

    # Save figure
    plt.tight_layout(pad=2.5, w_pad=1, h_pad=1.5)
    plt.savefig(file_name)
    plt.close()


def grouped_box_plot(y_lst, plot_title, legend, file_name, positions=[[1, 4, 7], [2, 5, 8]],
                     x_labels=['pd', 'perm', 'shap'], y_label='influence', fig_size=(8.0, 8.0)):
    # Create figure
    fig, ax = plt.subplots(len(y_lst[0]), 1, figsize=fig_size)

    # Plot layout
    ax_settings(ax.flatten(), n_row=len(y_lst[0]), n_col=1, sub_titles=['Feature 1', 'Feature 2', 'Feature 3'],
                grid_axis='y')

    # For legends
    dictionary = {}

    # Iterate through subplots
    for row in range(len(y_lst[0])):
        # Get the data corresponding to subplot
        lst_data = list(map(lambda x: x[row], y_lst))

        # Create box plots for each dataset in lst_data
        for col in range(len(lst_data)):
            color = list(get_default_colors().values())[col]
            transparent_color = list(get_default_colors().values())[col] + [0.8]
            dictionary['bf{}'.format(col)] = ax[row].boxplot(lst_data[col], positions=positions[col], patch_artist=True,
                                                             widths=0.6,
                                                             boxprops=dict(facecolor=transparent_color, color=color),
                                                             capprops=dict(color=color), whiskerprops=dict(color=color),
                                                             flierprops=dict(color=color, markeredgecolor=color),
                                                             medianprops=dict(color=color))

    # Create legend in bottom of figure
    fig.legend(list(map(lambda x: x["boxes"][0], list(dictionary.values()))), legend, bbox_to_anchor=[0.54, 0.94],
               loc='center', prop={'size': 9}, ncol=len(positions))

    # Create plot titles and labels
    fig.suptitle(plot_title, x=0.55, y=0.99, fontsize=13, ha='center', fontname='serif')
    ax[len(y_lst[0]) - 1].set_xticks(np.array(positions[0]) + 0.5 * (len(positions) - 1), x_labels, fontsize=10,
                                     fontname='serif')
    fig.text(s=y_label, x=0.015, y=0.5, fontsize=10, ha='center', va='center', rotation='vertical',
             fontname='serif')

    # Make layout tight
    plt.tight_layout(pad=2, w_pad=1.2, h_pad=1.5)

    # Save plot as file_name
    plt.savefig(file_name)
    plt.close()


def linreg_grouped_box_plot(y_lst, plot_title, legend, positions, file_name):
    grouped_box_plot(y_lst=y_lst, plot_title=plot_title, x_labels=['pd', 'perm', 'shap'] + ['real influence'],
                     legend=legend, positions=positions, file_name=file_name)
