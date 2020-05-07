import pandas as pd
import numpy as np
import argparse
import time
import os
from pathlib import Path

import sklearn

from joblib import dump, load


def model_predict(args: argparse.Namespace) -> None:
    """

    :param args: path/to/data/file.xlsx. Must be unlabeled data.
    :return: feature importance, fit model, gridsearch results, and data transform mask.
    """

    rf = load('../model_checkpoints/model_checkpoint_wCalphad_gs_2019-09-17-02-09.joblib')
    # output_direc = '../HEA_alloys/wCALPHAD/Feature_set_2/'

    input_ = args.input

    transform_df = pd.read_csv('../transform_mask/FILENAME_HERE.csv')

    p = Path(input_)
    p = p.parent
    p = p.parent

    new_predictions = p / 'new_predictions'
    if not new_predictions.exists():
        new_predictions.mkdir()

    df_orig = pd.read_excel(input_)
    # print(df_orig)

    orig = df_orig.as_matrix()[:, 1:]

    alloy_id = pd.DataFrame(df_orig['Name'])
    # print(alloy_id)

    whereNan = np.isnan(list(orig[:, -1]))

    news = orig[whereNan]

    X_test = news[:, :-1]
    X_test = X_test[:, transform_df['0'].as_matrix()]

    predictions = rf.predict(X_test)
    # print(predictions)
    df_pred = pd.DataFrame(predictions)
    # print(df_pred)

    df_join = alloy_id.merge(df_pred, left_index=True, right_index=True)
    # df_join = df_join.rename(index=str, columns={'Name': 'Composition', '0': 'Predicted_EFA'})
    print(df_join)

    # prediction_csv = pd.DataFrame.to_csv(df_join, columns=['predictions']).to_csv('prediction.csv')
    df_join.to_csv(str(new_predictions) + '/FILENAME_HERE_' + str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='path/to/data/file.xlsx')

    return parser.parse_args()


def main():
    args = parser()
    model_predict(args)


if __name__ == '__main__':
    main()
