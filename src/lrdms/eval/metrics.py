import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, r2_score, roc_auc_score
from loguru import logger


def spearman(*, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Spearman's rank correlation coefficient.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.

    Returns:
        float: Spearman's rank correlation coefficient.
    """
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation


def ndcg(*, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Reference:
     - https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.

    Returns:
        float: NDCG score.
    """
    y_true_normalized = (y_true - y_true.mean()) / y_true.std()
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))


def topk_mean(*, y_pred: np.ndarray, y_true: np.ndarray, topk: int = 96) -> float:
    """
    Compute the mean of the top k true scores based on predicted scores.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        topk (int): Number of top scores to consider. Defaults to 96, which is the
            size of a 96-well plate.

    Returns:
        float: Mean of the top k true scores.
    """
    if len(y_true) < topk:
        return np.nan
    select_topk = np.argsort(y_pred)[-topk:]
    return np.mean(y_true[select_topk])


def r2(*, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the R^2 (coefficient of determination) regression score.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.

    Returns:
        float: R^2 score.
    """
    return r2_score(y_true, y_pred)


def hit_rate(*, y_pred: np.ndarray, y_true: np.ndarray, y_ref: float = 0.0, topk: int = 96) -> float:
    """
    Compute the hit rate at top k.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        y_ref (float): Reference value to compare against. Defaults to `0.0`, which
            is the wildtype (WT) for log-transformed WT-normalised fitness scores.
        topk (int): Number of top scores to consider. Defaults to 96, which is the
            size of a 96-well plate.

    Returns:
        float: Hit rate at top k.
    """
    n_above = np.sum(y_true[np.argsort(y_pred)[-topk:]] > y_ref)
    return float(n_above) / float(topk)


def aucroc(*, y_pred: np.ndarray, y_true: np.ndarray, y_cutoff: float) -> float:
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        y_cutoff (float): Cutoff value to binarize the true scores.

    Returns:
        float: ROC AUC score.
    """
    y_true_bin = y_true >= y_cutoff
    return roc_auc_score(y_true_bin, y_pred, average="micro")


def get_spearman_fractions(
    *, y_pred: np.ndarray, y_true: np.ndarray, spearman_fractions: np.ndarray = np.linspace(0.1, 1.0, 10)
) -> np.ndarray:
    """
    Compute Spearman's rank correlation coefficient for different fractions of the data.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        spearman_fractions (np.ndarray): Array of fractions to consider. Defaults to
            `np.linspace(0.1, 1.0, 10)`.

    Returns:
        np.ndarray: Array of Spearman's rank correlation coefficients for different fractions.
    """
    results = np.zeros(len(spearman_fractions))
    for i, f in enumerate(spearman_fractions):
        k = int(f * len(y_true))
        idx = np.argsort(y_true)[-k:]
        results[i] = spearmanr(y_pred[idx], y_true[idx]).correlation
    return results


def wt_improvement_metric(*, y_pred: np.ndarray, y_true: np.ndarray, y_wt: float, topk: int = 96) -> float:
    """
    Compute the weighted improvement metric.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        y_wt (float): Weight threshold.
        topk (int): Number of top scores to consider. Defaults to 96, which is the
            size of a 96-well plate.

    Returns:
        float: Weighted improvement metric.
    """
    hr = hit_rate(y_pred=y_pred, y_true=y_true, y_ref=y_wt, topk=topk)
    baseline = float(np.sum(y_true > y_wt)) / len(y_true)
    return hr / baseline


def topk_median(*, y_pred: np.ndarray, y_true: np.ndarray, topk: int = 96) -> float:
    """
    Compute the median of the top k true scores based on predicted scores.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        topk (int): Number of top scores to consider. Defaults to 96, which is the
            size of a 96-well plate.

    Returns:
        float: Median of the top k true scores.
    """
    if len(y_true) < topk:
        return np.nan
    select_topk = np.argsort(y_pred)[-topk:]
    return np.median(y_true[select_topk])


def topk_recall_from_quantile_q(*, y_pred: np.ndarray, y_true: np.ndarray, q: float = 0.9, k: int = 10) -> float:
    """
    Compute the recall at k from a quantile q, i.e. answering the question:
    "How many of the top k predictions are in the top q quantile of the true values?"

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        q (float, optional): Quantile to consider. Defaults to 0.9.
        k (int, optional): Number of top scores to consider. Defaults to 10.

    Returns:
        float: Recall at k from quantile q.
    """
    # Return how many of the top 10 predictions are in the top 10% of the true values
    top_k_preds = np.argsort(y_pred)[::-1][:k]
    top_q_true = np.quantile(y_true, q)
    return np.sum(y_true[top_k_preds] >= top_q_true) / k


def topk_best_fitness(*, y_pred: np.ndarray, y_true: np.ndarray, k: int = 10) -> float:
    """
    Compute the best fitness in the top k predictions.

    Args:
        y_pred (np.ndarray): Array of predicted scores.
        y_true (np.ndarray): Array of true scores.
        k (int): Number of top scores to consider.

    Returns:
        float: Best fitness in the top k predictions.
    """
    # Return the best fitness in the top k predictions
    top_k_preds = np.argsort(y_pred)[::-1][:k]
    return np.max(y_true[top_k_preds])


def dcg_at_k(scores: np.ndarray, k: int, relevance: str = "exponential") -> float:
    """
    Compute Discounted Cumulative Gain (DCG) at k.

    Args:
        scores (np.ndarray): Array of scores.
        k (int): Number of top scores to consider.
        relevance (str): Type of relevance. Either "linear" or "exponential".
            With "linear" relevance, the true scores are assumed to be larger than 0.

    Returns:
        float: DCG@k score.
    """
    # Create an array of indices representing the rank positions
    ranks = np.arange(1, k + 1)

    # Compute the discounted gain for each position
    discounts = np.log2(ranks + 1)
    if relevance == "linear":
        assert np.all(scores > 0), "Scores must be larger than 0 for linear relevance."
        gains = scores / discounts
    elif relevance == "exponential":
        gains = np.clip((np.power(2, scores) - 1), a_min=0.0, a_max=np.inf) / discounts
    else:
        raise ValueError("Relevance must be either 'linear' or 'exponential'")

    # Return the sum of gains which is the DCG@k
    return np.sum(gains)


def ndcg_at_k(*, y_pred: np.ndarray, y_true: np.ndarray, k: int, relevance: str = "exponential") -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        y_true (np.ndarray): Array of true scores.
        y_pred (np.ndarray): Array of predicted scores.
        k (int): Number of top scores to consider.
        relevance (str): Type of relevance. Either "linear" or "exponential".
            With "linear" relevance, the true scores are assumed to be larger than 0 and the results
            are the same as the scikit-learn `ndcg_score` function.

    Returns:
        float: NDCG@k score.
    """
    # Calculate DCG@k for the predicted and ideal scores
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    if len(top_k_indices) < k:
        logger.warning(f"Not enough scores to calculate NDCG@k: {len(top_k_indices)} < {k}. Returning NaN.")
        return np.nan

    # Calculate the actual discounted cumulative gain
    scores_at_k = y_true[top_k_indices]
    dcg = dcg_at_k(scores_at_k, k=k, relevance=relevance)

    # Calculate the ideal discounted cumulative gain
    ideal_scores_at_k = np.sort(y_true)[::-1][:k]
    idcg = dcg_at_k(ideal_scores_at_k, k=k, relevance=relevance)

    # Return the NDCG@k
    return 0 if idcg == 0 else dcg / idcg


def precision_at_k(*, y_pred: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """
    Compute Precision at k.

    Args:
        y_true (np.ndarray): Array of true scores.
        y_pred (np.ndarray): Array of predicted scores.
        k (int): Number of top scores to consider.

    Returns:
        float: Precision@k score.
    """
    # Calculate the top k predictions
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    if len(top_k_indices) < k:
        logger.warning(f"Not enough scores to calculate Precision@k: {len(top_k_indices)} < {k}. Returning NaN.")
        return np.nan

    top_k_predictions = y_true[top_k_indices] > 0.0

    # Calculate the precision
    precision = np.mean(top_k_predictions)

    # Return the precision@k
    return precision
