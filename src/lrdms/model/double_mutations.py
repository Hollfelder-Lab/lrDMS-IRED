import numpy as np
import pandas as pd
import xgboost as xgb
import re
from typing import Literal

from lrdms.features.dms import CombinabilityFeaturiser


def _reduce_grouped_features(X: pd.DataFrame, reduce_funcs: list[callable]) -> pd.DataFrame:
    """Reduce a group of features by applying a reduce function to each group."""

    # Get number of single-group count from highets column suffix
    n_single_groups = int(X.columns.str.extract(r"_(\d+)$").dropna().max().values[0])
    # Get single group
    single_groups = np.array(
        [sorted([col for col in X.columns if re.search(rf"_{i+1}$", col)]) for i in range(n_single_groups)]
    )
    # Get all columns that don't have a number suffix
    combined_groups = [col for col in X.columns if not re.search(r"_\d+$", col)]
    # For each reduce function, apply to each single group
    df = X[combined_groups].copy()
    for reduce in reduce_funcs:
        for feature in single_groups.T:
            feature_name = feature[0].rsplit("_", 1)[0] + f"_{reduce.__name__}"
            df.loc[:, feature_name] = reduce(X[feature], axis=1)
    return df


class CombinabilityModel:
    def __init__(
        self,
        model: xgb.XGBModel,
        wt_seq_len: int,
        features_to_use: list[str],
        aggr_funcs=[np.mean, np.max, np.min],
        combinability_version: Literal[1, 2] = 1,
        **combinability_kwargs,
    ):
        self.model = model
        self.wt_seq_len = wt_seq_len
        self.aggr_funcs = aggr_funcs
        self.combinability_featuriser = CombinabilityFeaturiser(
            seq_len=wt_seq_len, version=combinability_version, **combinability_kwargs
        )
        self.features_to_use = features_to_use

    def prepare_features(self, X):
        # Add combinability features to X
        X_featurised = self.combinability_featuriser.transform(X)
        X_featurised = _reduce_grouped_features(X_featurised, self.aggr_funcs)

        # Add esm_epi feature
        if "esm_epi" in self.features_to_use:
            X_featurised["esm_epi"] = (
                X_featurised["esm_fitness"] - X_featurised["esm_fitness_min"] - X_featurised["esm_fitness_max"]
            )
        return X_featurised[self.features_to_use]

    def fit(self, X, y, combinability_data, **model_fit_kwargs):
        # Fit combinability features
        self.combinability_featuriser.fit(combinability_data)

        # Prepare features
        X_featurised = self.prepare_features(X)

        # PATCH: Featurise eval_set
        if "eval_set" in model_fit_kwargs:
            _updated_eval_set = []
            for eval_set in model_fit_kwargs["eval_set"]:
                X_eval, y_eval = eval_set
                X_eval_featurised = self.prepare_features(X_eval)
                _updated_eval_set.append((X_eval_featurised, y_eval))
            model_fit_kwargs["eval_set"] = _updated_eval_set
        ### END PATCH

        # Fit model
        self.model.fit(X_featurised, y, **model_fit_kwargs)

    def predict(self, X):
        # Prepare features
        X_featurised = self.prepare_features(X)

        # Predict
        return self.model.predict(X_featurised)
