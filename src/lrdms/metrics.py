import numpy as np

from loguru import logger


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


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, relevance: str = "exponential") -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        y_true (np.ndarray): Array of true scores.
        y_pred (np.ndarray): Array of predicted scores.
        k (int): Number of top scores to consider.
        relevance (str): Type of relevance. Either "linear" or "exponential".
            With "linear" relevance, the true scores are assumed to be larger than 0 and the results
            are the same as the scikit-learn ndcg_score function.

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

    # Avoid zero division
    if idcg == 0:
        return 0

    # Return the NDCG@k
    return dcg / idcg


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
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
