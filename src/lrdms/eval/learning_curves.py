# Well plate sizes
from typing import Callable
import numpy as np

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from lrdms.utils.common import exists

from loguru import logger


WELL_PLATE_SIZES = [3, 6, 12, 24, 48, 96, 2 * 96, 3 * 96, 384, 2 * 384, 3 * 384, 1536]


def compute_learning_curves(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    metrics: Callable | list[Callable],
    sizes: list[int | float] = WELL_PLATE_SIZES,
    cv=KFold(n_splits=10, shuffle=True, random_state=7),
    shuffle_data: bool = True,
    random_state: int = 7,
    X_wt=None,
    y_wt=None,
    store_predictions: bool = False,
    stratify_bins: int | None = 10,
):
    output = {}

    # If groups is an integer N and y continuous, then bin y into N groups
    if exists(stratify_bins) and (y.dtype in [np.float32, np.float64]):
        assert stratify_bins > 1
        y_bins = np.digitize(y, bins=np.linspace(y.min(), y.max(), stratify_bins))
    else:
        y_bins = y

    cv_iter = list(cv.split(X, y=y_bins))
    cv_subset_size = [len(cv_train_subset) for cv_train_subset, _ in cv_iter]
    cv_min_size = min(cv_subset_size)

    if cv_min_size < max(sizes):
        logger.warning(f"Dropping sizes > {cv_min_size} (smallest cross-validation subset size)")
        sizes = [s for s in sizes if s <= cv_min_size]
        sizes += [cv_min_size]

    if shuffle_data:
        X, y = shuffle(X, y, random_state=random_state)  # returns shuffled copies

    if not isinstance(metrics, list):
        metrics = [metrics]

    # Outer loop: iterate over the cross-validation splits
    for cv_idx, (train_idxs, test_idxs) in enumerate(cv_iter):
        output[f"cv{cv_idx}"] = {}
        cv_result = output[f"cv{cv_idx}"]
        cv_result["train_size"] = len(train_idxs) + len(test_idxs)
        cv_result["test_size"] = len(test_idxs)

        # Inner loop: iterate over the different dataset sizes
        for size in sizes:
            cv_result[f"{size}"] = {}
            cv_result["train_propotion"] = len(train_idxs) / (len(train_idxs) + len(test_idxs))

            # Get the training and test sets
            X_train, X_test = X[train_idxs[:size]], X[test_idxs]
            y_train, y_test = y[train_idxs[:size]], y[test_idxs]

            # Append the wildtype data to the training set (we assume WT is always known)
            if exists(X_wt) and exists(y_wt):
                X_train = np.concatenate([X_train, np.atleast_1d(X_wt)])
                y_train = np.concatenate([y_train, np.atleast_1d(y_wt)])

            # Fit the model
            model.fit(X_train, y_train)

            # Compute the predictions
            y_pred = model.predict(X_test)

            if store_predictions:
                cv_result[f"{size}"]["y_pred"] = y_pred
                cv_result[f"{size}"]["y_test"] = y_test

            # Compute the metrics
            for metric in metrics:
                cv_result[f"{size}"][metric.__name__] = metric(y_true=y_test, y_pred=y_pred)

    # Average over the cross-validation splits
    output["result_stats"] = {}
    for size in sizes:
        output["result_stats"][f"{size}"] = {}
        result_stats = output["result_stats"][f"{size}"]
        for metric in metrics:
            result_stats[metric.__name__] = {}
            result_stats_for_metric = result_stats[metric.__name__]
            result_stats_for_metric["mean"] = np.mean(
                [output[f"cv{i}"][f"{size}"][metric.__name__] for i in range(cv.get_n_splits())]
            )
            result_stats_for_metric["std"] = np.std(
                [output[f"cv{i}"][f"{size}"][metric.__name__] for i in range(cv.get_n_splits())]
            )
            result_stats_for_metric["med"] = np.median(
                [output[f"cv{i}"][f"{size}"][metric.__name__] for i in range(cv.get_n_splits())]
            )
            result_stats_for_metric["q25"] = np.quantile(
                [output[f"cv{i}"][f"{size}"][metric.__name__] for i in range(cv.get_n_splits())], 0.25
            )
            result_stats_for_metric["q75"] = np.quantile(
                [output[f"cv{i}"][f"{size}"][metric.__name__] for i in range(cv.get_n_splits())], 0.75
            )

    return output
