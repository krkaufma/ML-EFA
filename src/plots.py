import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from pathlib import Path
import argparse


def main(args: argparse.Namespace) -> None:
    """

    :param args: path/to/data/file.csv with known and predicted values.

    :return: specified plots
    """

    load_file = args.input

    df = pd.read_csv(load_file)

    p = Path(load_file)
    p = p.parent
    p = p.parent

    plots = p / 'plots'
    if not plots.exists():
        plots.mkdir()

    # Linear plot of all predictions
    dims = (7, 7)
    plt.rcParams['figure.figsize'] = dims
    df_preds = df['Predicted_EFA']
    df_truth = df['True_Label']
    linear_a = sns.regplot(x=df_truth, y=df_preds, fit_reg=False)
    linear_a.set_xlim(0, 150)
    linear_a.set_ylim(0, 150)
    plt.plot([0, 150], [0, 150], linewidth=2, color="b")
    # plt.title(file)
    plt.savefig(str(plots) + '/Orig_56_Predicted_EFA_no_Calphad_V2_updated_'
                + str(time.strftime("%Y-%m-%d-%I-%M")) + '.png')
    plt.show()

    # Linear plots with different groups/colors for new data
    dims = (7, 7)
    plt.rcParams['figure.figsize'] = dims

    df_preds_orig = df['Predicted_EFA'].loc[df['Group'] == 1]
    df_truth_orig = df['True_Label'].loc[df['Group'] == 1]

    df_preds_new = df['Predicted_EFA'].loc[df['Group'] == 2]
    df_truth_new = df['True_Label'].loc[df['Group'] == 2]

    linear_m = sns.regplot(x=df_truth_orig, y=df_preds_orig, fit_reg=False, color="b", marker='.')
    linear_m = sns.regplot(x=df_truth_new, y=df_preds_new, fit_reg=False, color="r", marker='.')
    linear_m.set_xlim(0, 150)
    linear_m.set_ylim(0, 150)
    plt.plot([0, 150], [0, 150], linewidth=2, color="b")
    plt.savefig(str(plots) + '/FILENAME_HERE_' + str(time.strftime("%Y-%m-%d-%I-%M")) + '.png')
    plt.show()


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='path/to/data/file.csv')

    return parser.parse_args()


args = parser()
main(args)
