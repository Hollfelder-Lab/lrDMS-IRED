"""
Adapted from https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/tree/main.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from tqdm.autonotebook import tqdm

from lrdms.constants import DATA_PATH
from lrdms.eval.metrics import hit_rate
from lrdms.features.tokenizer import SequenceTokenizer
from abc import ABC, abstractmethod
from lrdms.utils.common import default
from typing import Sequence, Literal

from loguru import logger


class BasePredictor(ABC):
    """Abstract class for predictors to implement a common API for all predictors.

    Subclasses should implement the `seq2feat`, `fit`, and `predict` methods.
    """

    def __init__(self, **kwargs):
        pass

    def __repr__(self) -> str:
        _reg_coef = self.reg_coef if isinstance(self.reg_coef, str) else f"{self.reg_coef:.1e}"
        return f"{self.__class__.__name__}(reg_coef={_reg_coef}) at {hex(id(self))}"

    @abstractmethod
    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        """Converts a list of sequences to a 2D feature matrix of shape [b, f]"""
        pass

    @abstractmethod
    def fit(self, seqs: Sequence[str], labels: np.ndarray) -> None:
        """Trains the model"""
        pass

    @abstractmethod
    def predict(self, seqs: Sequence[str]) -> np.ndarray:
        """Gets model predictions"""
        pass


class BaseRegressionPredictor(BasePredictor, ABC):
    """Base class for regression predictors.

    Subclasses should implement the `seq2feat` method to convert sequences to features.
    """

    REG_COEF_SEARCH_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]  # Default search list when reg_coef is set to "CV"
    N_CV_SPLITS = 5

    def __init__(self, reg_coef: float | Literal["CV"] | None = None, linear_model_cls=Ridge, **kwargs):
        self.reg_coef = reg_coef
        self.linear_model_cls = linear_model_cls
        self.model = None

    def fit(self, seqs: Sequence[str], labels: np.ndarray) -> None:
        X = self.seq2feat(seqs)
        assert X.shape[0] == len(labels), "Number of labels must match number of sequences"
        assert X.ndim == 2, "Features must be 2D: [b, f]"

        # Find the best reg coef using cross validation if not set
        if self.reg_coef is None or self.reg_coef == "CV":
            logger.debug("Cross validating reg coef...")
            best_rc, best_score = None, -np.inf

            for rc in tqdm(self.REG_COEF_SEARCH_LIST, desc="CV to find best reg coef", leave=False):
                logger.debug(f"Cross validating reg coef {rc}")
                model = self.linear_model_cls(alpha=rc)
                score = cross_val_score(
                    model, X, labels, cv=self.N_CV_SPLITS, scoring=make_scorer(hit_rate, topk=10)
                ).mean()

                if score > best_score:
                    best_rc = rc
                    best_score = score

            logger.info(f"Best CV reg coef {best_rc} with score {best_score:.3f}")
            self.reg_coef = best_rc

        # Train the model
        self.model = self.linear_model_cls(alpha=self.reg_coef)
        self.model.fit(X, labels)

    def predict(self, seqs: Sequence[str]) -> np.ndarray:
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        # Featurise the data
        X = self.seq2feat(seqs)
        return self.model.predict(X)


class JointPredictor(BaseRegressionPredictor):
    """Predictor to combine multiple regression predictors by training jointly."""

    def __init__(self, predictors: Sequence[BaseRegressionPredictor], reg_coef="CV", **kwargs):
        super().__init__(reg_coef, **kwargs)
        self.predictors = predictors

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        # To apply different regularziation coefficients we scale the features
        # by a multiplier in Ridge regression
        features = [p.seq2feat(seqs) * np.sqrt(1.0 / p.reg_coef) for p in self.predictors]
        return np.concatenate(features, axis=1)


class RandomShufflePredictor(BasePredictor):
    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        return np.zeros((len(seqs), 1))

    def fit(self, seqs: Sequence[str], labels: np.ndarray) -> None:
        self.train_labels = labels

    def predict(self, seqs: Sequence[str]) -> np.ndarray:
        return np.random.choice(self.train_labels, size=len(seqs), replace=True)


class MutationRadiusPredictor(BaseRegressionPredictor):
    def __init__(self, wt, reg_coef: float | Literal["CV"] | None = None, **kwargs):
        super().__init__(reg_coef, **kwargs)
        self.wt = wt

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        mutation_counts = np.zeros(len(seqs))
        for i, s in enumerate(seqs):
            for j in range(len(self.wt)):
                if self.wt[j] != s[j]:
                    mutation_counts[i] += 1
        return -mutation_counts[:, None]

    def fit(self, *args, **kwargs):
        pass

    def predict(self, seqs: Sequence[str]) -> np.ndarray:
        return self.seq2feat(seqs).squeeze()


class AAIndexPredictor(BaseRegressionPredictor):
    """AAIndex encoding + ridge regression."""

    def __init__(
        self,
        reg_coef=1.0,
        n_components: int = 19,
        seed: int = 7,
        out_dir: Path = DATA_PATH,
        **kwargs,
    ):
        super().__init__(reg_coef, linear_model_cls=Ridge, **kwargs)
        from lrdms.features.aaindex import generate_aaindex_pca_embeddings

        self.n_components = n_components
        self.seed = seed
        self.aa_matrix = generate_aaindex_pca_embeddings(out_dir=out_dir, n_components=n_components, seed=seed)
        self.tokenizer = SequenceTokenizer()

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        tokenized_seqs = self.tokenizer.encode(seqs, to_one_hot=False)
        emb_seqs = self.aa_matrix[tokenized_seqs]  # [b, l, c]
        return emb_seqs.reshape(emb_seqs.shape[0], -1)  # [b, l * c]


class SubstitutionMatrixPredictor(BaseRegressionPredictor):
    """Substitution matrix encoding + ridge regression."""

    def __init__(self, wt: str, reg_coef=1e-8, substitution_matrix: str = "BLOSUM62", **kwargs):
        super().__init__(reg_coef, **kwargs)
        self.wt = wt
        self.substitution_matrix = substitution_matrix
        from Bio.Align import substitution_matrices

        self.matrix = substitution_matrices.load(substitution_matrix)
        self.alphabet = self.matrix.alphabet
        for i, c in enumerate(self.wt):
            assert c in self.alphabet, f"unexpected AA {c} (pos {i})"

    @staticmethod
    def _get_scores(seqs: Sequence[str], wt: str, substitution_matrix) -> np.ndarray:
        scores = np.zeros(len(seqs))
        wt_score = 0
        for j in range(len(wt)):
            wt_score += substitution_matrix[wt[j], wt[j]]
        for i, s in enumerate(seqs):
            for j in range(len(wt)):
                if s[j] not in substitution_matrix.alphabet:
                    print(f"unexpected AA {s[j]} (seq {i}, pos {j})")
                scores[i] += substitution_matrix[wt[j], s[j]]
        return scores - wt_score

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        return self._get_scores(seqs, self.wt, self.matrix)[:, None]

    def fit(self, *args, **kwargs):
        pass

    def predict(self, seqs: Sequence[str]) -> np.ndarray:
        return self.seq2feat(seqs).squeeze()


class OnehotRidgePredictor(BaseRegressionPredictor):
    """Simple one hot encoding + ridge regression."""

    def __init__(self, reg_coef=1.0, **kwargs):
        super().__init__(reg_coef, Ridge, **kwargs)
        self.tokenizer = SequenceTokenizer()

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        # seqs: [b, l]

        # Featurise the data  # TODO: Clean up these functions
        feats = self.tokenizer(seqs, to_one_hot=True).numpy()  # [b l v]

        # ... flatten X to be 2D for linear regression
        feats = feats.reshape(feats.shape[0], -1)  # [b l*v]
        return feats


class LookUpPredictor(BaseRegressionPredictor):
    """Simple lookup predictor for pre-computed values."""

    def __init__(
        self,
        seq2score_dict: dict[str, float],
        reg_coef: float | Literal["CV"] | None = 1e-8,
        default_val: float = -np.inf,
        **kwargs,
    ):
        """Initialise the lookup predictor.

        Args:
            - seq2score_dict (dict): A dictionary mapping sequences to scores.
            - reg_coef (float): Regularization coefficient for Ridge regression.
            - default_val (float): Default value to use for sequences not in the dict.
        """
        super().__init__(reg_coef, Ridge)
        self.seq2score_dict = seq2score_dict
        self.default = default_val

    def seq2feat(self, seqs: Sequence[str], default_val: float | None = None) -> np.ndarray:
        default_val = default(default_val, self.default)
        return np.array([self.seq2score_dict.get(s, default_val) for s in seqs])[:, None]

    def fit(self, *args, **kwargs):
        # No training needed for lookup
        pass

    def predict(self, seqs: Sequence[str]) -> np.ndarray:
        return self.seq2feat(seqs).squeeze()
