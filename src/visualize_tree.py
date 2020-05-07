import os
import sys
import graphviz
import six
import pydot
import pandas as pd
import argparse
from subprocess import check_call
from itertools import compress
from sklearn import tree
from sklearn.tree import export_graphviz
from joblib import dump, load

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
sys.path.append('C:/Program Files (x86)/Graphviz2.38/bin/')

model_path = '../model_checkpoints/model_checkpoint_wCalphad_gs_2019-09-17-02-09.joblib'

estimator = load(model_path)

dotfile = six.StringIO()


def visualize(args: argparse.Namespace, i_tree=0) -> None:
    """

    :param args: path/to/data/file.xlsx.
    :return: feature importance, fit model, gridsearch results, and data transform mask.
    """

    input_ = args.input
    df_orig = pd.read_excel(input_)
    col = list(df_orig.columns)[1:-1]

    # Initial feature elimination if you have a predetermined mask
    transform_first = True
    if transform_first is True:
        transform_mask_init = pd.read_csv('../transform_mask/FILENAME_HERE.csv')
        truth_series = pd.Series(transform_mask_init['0'], name='bools')
        df_orig.drop(['Name', 'EFA'], axis=1, inplace=True)
        df_orig = pd.DataFrame(df_orig.iloc[:, truth_series.values])

        col = list(df_orig.columns)[:]

    dotfile = six.StringIO()  # Do NOT delete
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    sys.path.append('C:/Program Files (x86)/Graphviz2.38/bin/')

    for tree_in_forest in estimator:
        export_graphviz(tree_in_forest, out_file='tree.dot',
                        feature_names=col,
                        filled=True,
                        rounded=True)
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        name = 'tree' + str(i_tree)
        print('Now exporting: ' + name)
        check_call(['dot', '-Tpng', 'tree.dot', '-o', name + '.png'])

        i_tree += 1


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='path/to/data/file.xlsx')

    return parser.parse_args()


def main():
    args = parser()
    visualize(args)


if __name__ == '__main__':
    main()
