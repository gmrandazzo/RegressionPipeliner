"""
boostedtrees.py

Copyright (C) <2023>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import xgboost as xgb
import catboost as catb
from .dataset import Split

def xgbreg(split : Split):
    """
    XGBoost Regression Function Documentation

    Parameters:
        - split (Split): An instance of the `Split` class containing
          training, testing, and validation data splits.

    Returns:
        - predictions (List[float]): A list of predicted target values for the validation dataset.

    Usage:
        This function applies XGBoost regression to the given data split.
        It uses a predefined set of hyperparameters (hp) to create an XGBoost Regressor model.
        The model is trained on the training data and evaluated
        on the validation data to monitor performance.
        Finally, it returns the predicted target values for the validation dataset.

    Example:
        # Import the necessary modules and create a Split object
        from mymodule import xgbreg, Split

        data_split = Split()
        data_split.x_train = training_features
        data_split.y_train = training_labels
        data_split.x_test = testing_features
        data_split.y_test = testing_labels
        data_split.x_val = validation_features
        data_split.y_val = validation_labels

        # Perform XGBoost regression
        xgboost_predictions = xgbreg(data_split)
    """
    hyp_params = {'base_score': 0.5,
                  'colsample_bylevel': 1,
                  'colsample_bytree': 0.66,
                  'gamma': 0,
                  'learning_rate': 0.05,
                  'max_delta_step': 1,
                  'max_depth': 5,
                  'min_child_weight': 5,
                  'n_estimators': 3000,
                  'reg_alpha': 0,
                  'reg_lambda': 1,
                  'scale_pos_weight': 1,
                  'subsample': 0.53}
    reg = xgb.XGBRegressor(**hyp_params)
    reg.fit(split.x_train, split.y_train,
            eval_set=[(split.x_test, split.y_test)],
            verbose=0)
    return reg.predict(split.x_val)


def catbreg(split : Split):
    """
    CatBoost Regression Function Documentation

    Parameters:
        - split (Split): An instance of the `Split` class containing
          training, testing, and validation data splits.

    Returns:
        - predictions (List[float]): A list of predicted target values for the validation dataset.

    Usage:
        This function applies CatBoost regression to the given data split.
        It uses a predefined set of hyperparameters (hp) to create a CatBoost Regressor model.
        The model is trained on the training data and evaluated
        on the validation data to monitor performance.
        Finally, it returns the predicted target values for the validation dataset.

    Example:
        # Import the necessary modules and create a Split object
        from mymodule import catbreg, Split

        data_split = Split()
        data_split.x_train = training_features
        data_split.y_train = training_labels
        data_split.x_test = testing_features
        data_split.y_test = testing_labels
        data_split.x_val = validation_features
        data_split.y_val = validation_labels

        # Perform CatBoost regression
        catboost_predictions = catbreg(data_split)
    """
    hyp_params = {'iterations': 3000,
                  'learning_rate': 0.05,
                  'depth': 5,
                  'nan_mode': 'Forbidden',
                  'use_best_model': True,
                  'allow_const_label': True}
    reg = catb.CatBoostRegressor(**hyp_params)
    reg.fit(split.x_train, split.y_train,
            eval_set=[(split.x_test, split.y_test)])
    return reg.predict(split.x_val)
