import pandas as pd
import numpy as np
import argparse
import os
import rfpimp
import sklearn
import time
from sklearn import feature_selection
from joblib import dump, load
from pathlib import Path


def get_importance(args: argparse.Namespace) -> None:
    """

    Source: https://explained.ai/rf-importance/index.html

    :param args: path/to/data/file.xlsx
    :return: feature importance, fit model, gridsearch results, and data transform mask.
    """

    select_from_model = False
    transform_first = False

    input_ = args.input

    p = Path(input_)
    p = p.parent
    p = p.parent

    importance = p / 'importance'
    model_checkpoints = p / 'model_checkpoints'
    rf_best_params = p / 'rf_best_params'
    transform_mask = p / 'transform_mask'

    if not importance.exists():
        importance.mkdir()

    if not model_checkpoints.exists():
        model_checkpoints.mkdir()

    if not rf_best_params.exists():
        rf_best_params.mkdir()

    if not transform_mask.exists():
        transform_mask.mkdir()

    df_orig = pd.read_excel(input_)

    orig = df_orig.as_matrix()[:, 1:]

    feature_names = list(df_orig.columns)[1:-1]

    whereNan = np.isnan(list(orig[:, -1]))

    olds = orig[np.logical_not(whereNan)]

    news = orig[whereNan]

    y_train = olds[:, -1]
    X_train = olds[:, :-1]

    X_test = news[:, :-1]

    Xdf = pd.DataFrame(X_train, columns=feature_names)
    ydf = pd.Series(y_train)

    # Initial feature elimination if you have a predetermined mask
    if transform_first is True:
        transform_mask_init = pd.read_csv('../transform_mask/Transform_FILENAME_HERE.csv')
        X_train = X_train[:, transform_mask_init['0'].as_matrix()]

        print("The initially masked Xdf is shape: ")
        print(X_train.shape)

        truth_series = pd.Series(transform_mask_init['0'], name='bools')
        Xdf = pd.DataFrame(Xdf.iloc[:, truth_series.values])

        # save_new_df = pd.DataFrame(X_train)
        # Xdf.to_excel("test_new_cols_1.xlsx")
        # save_new_df.to_excel("test_1.xlsx")

    # Feature elimination based on importance and Select From Model method
    if select_from_model is True:
        print("Selecting the best features in your dataset.")
        rf = sklearn.ensemble.RandomForestRegressor(n_jobs=-1, random_state=42, bootstrap=True,
                                                    n_estimators=2000,
                                                    max_features=0.5)

        print("The original Xdf is shape: ")
        print(X_train.shape)

        select_fm = sklearn.feature_selection.SelectFromModel(estimator=rf, threshold=-np.inf, max_features=8)

        select_fm.fit_transform(X_train, y_train)

        feature_conds = select_fm.get_support()
        transform_df = pd.DataFrame(feature_conds)
        transform_df.to_csv(
            str(transform_mask) + "/Transform_FILENAME_HERE" + str(time.strftime("%Y-%m-%d-%I-%M")) + ".csv")
        X_train = select_fm.transform(X_train)

        print("Finished transforming the data; new xdf shape is: ")
        print(X_train.shape)

        Xdf = Xdf[Xdf.columns[feature_conds]]

    rf = sklearn.ensemble.RandomForestRegressor(n_jobs=-1, random_state=42, bootstrap=True)

    gs = sklearn.model_selection.GridSearchCV(rf, param_grid={'n_estimators': [i for i in range(10, 110, 10)],
                                                              'criterion': ['mse', 'mae'],
                                                              'max_features': [i for i in range(1, X_train.shape[1])]},
                                              scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, refit=True, verbose=1)

    print("Optimizing the Hyperparameters. Please be patient.")
    yay = gs.fit(X_train, y_train)

    grid_search_df = pd.DataFrame(gs.cv_results_)
    grid_search_df.to_csv(
        str(rf_best_params) + '/gridsearch_FILENAME_HERE_' + str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')
    best_results_df = pd.DataFrame(gs.best_params_, index=[0])
    best_results_df.to_csv(str(rf_best_params) + '/gridsearch_Calphad_FILENAME_HERE_best_params_' + str(
        time.strftime("%Y-%m-%d-%I-%M")) + '.csv')

    rf = sklearn.ensemble.RandomForestRegressor(**yay.best_params_, random_state=42, n_jobs=-1, bootstrap=True,
                                                verbose=0)

    print("Optimal Hyperparameters located. Fitting model to these parameters now.")
    rf.fit(X_train, y_train)

    imp = rfpimp.importances(rf, Xdf, ydf)

    viz = rfpimp.plot_importances(imp)
    viz.save(str(importance) + f'/importances_FILENAME_HERE_-{int(time.time())}.png')
    viz.view()

    dump(rf,
         str(model_checkpoints) + '/model_checkpoint_FILENAME_HERE_' + str(time.strftime("%Y-%m-%d-%I-%M")) + '.joblib')


def parser() -> argparse.Namespace:
    """
    args: path/to/data/file.xlsx
    :return: path/to/data/file.xlsx
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='path/to/data/file.xlsx')

    return parser.parse_args()


def main():
    args = parser()
    get_importance(args)


if __name__ == '__main__':
    main()
