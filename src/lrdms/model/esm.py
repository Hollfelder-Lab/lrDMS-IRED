from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import dill
import torch
from tqdm.autonotebook import tqdm
from loguru import logger
from typing import Callable, Literal, Union
import esm
import numpy.typing as npt


from lrdms.constants import DATA_PATH
from lrdms.utils.mutations import Variant, NATURAL_AA


class ESMEncoder:
    VALID_MODELS = tuple(
        [
            "esm1b_t33_650M_UR50S",
            "esm2_t6_8M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D",
            "esm2_t36_3B_UR50D",
            "esm2_t48_15B_UR50D",
        ]
    )

    def __init__(
        self,
        device: str = "cpu",
        pretrained_model: str = "esm2_t30_150M_UR50D",
        initialise_now: bool = False,
    ):
        if pretrained_model not in self.VALID_MODELS:
            raise ValueError(f"Invalid model name: {pretrained_model}. Valid names are: {self.VALID_MODELS}")
        self.pretrained_model = pretrained_model
        self.device = device
        if initialise_now:
            self.initialise()
        else:
            self.initialised = False

    def initialise(self):
        """Initialising can take a while, so we offer it as a separate step."""
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.pretrained_model)
        self.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # disables dropout for deterministic results
        self.initialised = True

    @property
    def n_tokens(self) -> int:
        return len(self.alphabet.all_toks)

    @property
    def name(self) -> str:
        return self.pretrained_model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pretrained_model={self.pretrained_model}, device={self.device})"

    def one_hot(self, aa_seqs: Sequence[str]) -> torch.Tensor:
        tokenised_aa_seqs = self.tokenize(aa_seqs)
        return torch.nn.functional.one_hot(tokenised_aa_seqs, self.n_tokens)

    def tokenize(self, aa_seqs: Sequence[str]) -> torch.Tensor:
        # add dummy labels
        aa_seqs_with_dummy_labels = [[None, seq] for seq in aa_seqs]
        _, _, batch_tokens = self.batch_converter(aa_seqs_with_dummy_labels)
        return batch_tokens  # [n_seqs, n_aa_res + 2]

    def __call__(
        self,
        aa_seqs: Union[str, Sequence[str]],
        drop_padding: bool = True,
        detach: bool = True,
        to_cpu: bool = False,
        to_numpy: bool = False,
        output: Literal["logits", "representations", "attentions", "contacts"] = "representations",
        reduce: Callable[[torch.Tensor, int], torch.Tensor] = None,
    ) -> Union[torch.Tensor, npt.NDArray]:
        if not self.initialised:
            self.initialise()
        if isinstance(aa_seqs, str):
            aa_seqs = [aa_seqs]
        batch_tokens = self.tokenize(aa_seqs).to(self.device)  # [n_seqs, n_aa_res + 2]
        logger.debug("Batch tokens shape: %s", str(batch_tokens.shape))

        # Extract per-residue representations
        logger.debug("Running model")
        with torch.inference_mode():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers], return_contacts=True)
            del batch_tokens

        # Select output
        if output in ["logits", "probabilities"]:
            out = results["logits"]  # [n_seqs, n_aa_res + 2, n_aa_res + 2]
            if output == "probabilities":
                out = torch.softmax(out, dim=-1)
        elif output == "contacts":
            return self._detach_cpu_numpy(results[output], detach, to_cpu, to_numpy)  # [n_seqs, n_aa_res, n_aa_res]
        elif output == "attentions":
            out = results[output]
            out = out[..., 1:-1, 1:-1] if drop_padding else out
            return self._detach_cpu_numpy(
                out, detach, to_cpu, to_numpy
            )  # [n_seqs, n_layers, n_heads, n_aa_res, n_aa_res]
        elif output == "representations":
            out = results[output][self.model.num_layers]  # [n_seqs, n_aa_res + 2, n_features]
        else:
            raise ValueError(f"Invalid output: {output}")
        logger.debug("Output shape: %s", str(out.shape))

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        #  The last token is always a end-of-sequence token. So for sequences of the same length
        #  the last residue is token -1.
        # TODO: This needs to be adapted for sequences of varying
        #  lengths to deal with the padding at the end of the sequence.
        out = out[:, 1:-1, :] if drop_padding else out
        out = reduce(out, dim=1) if reduce else out
        return self._detach_cpu_numpy(out, detach, to_cpu, to_numpy)  # [n_seqs, n_aa_res, n_features]

    @staticmethod
    def _detach_cpu_numpy(
        x: torch.Tensor, detach: bool, to_cpu: bool, to_numpy: bool
    ) -> Union[npt.NDArray, torch.Tensor]:
        x = x.detach() if detach else x
        x = x.cpu() if to_cpu else x
        x = x.cpu().numpy() if to_numpy else x
        return x

    def to(self, device: str) -> None:
        self.model.to(device)
        self.device = device

    @property
    def num_features(self) -> int:
        num_feats = {
            "esm1b_t33_650M_UR50S": 1280,
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
            "esm2_t48_15B_UR50D": 5120,
        }
        return num_feats[self.pretrained_model]


def to_batch(seq: Sequence[Any], batch_size: int) -> list[Sequence[Any]]:
    assert batch_size > 0
    batches = []
    start_index = 0
    while start_index < len(seq):
        batches.append(seq[start_index : start_index + batch_size])
        start_index += batch_size
    return batches


def apply_mask(seq: str, pos: Sequence[int], mask_token: str = "<mask>") -> str:
    if isinstance(pos, int):
        # Case: single position
        masked_seq = seq[:pos] + mask_token + seq[pos + 1 :]
    else:
        # Case: multiple positions
        masked_seq = list(seq)
        for p in pos:
            masked_seq[p] = mask_token
        masked_seq = "".join(masked_seq)
    return masked_seq


class MaskedLMFitnessEvaluator:
    """
    A class for evaluating the fitness of protein variants using a masked language model.
    Fitness is defined as the approximate pseudo log likelihood of the vairant minus the wildtype:
        fitness := PLL(variant) - PLL(wildtype)
                ~ sum_{i \in mutated_pos} \log{p(variant_i|variant_{-i})} - \log{p(wt_i|wt_{-i})})

    NOTE: Pseudo-log-likelihood (PLL) as defined in
        https://www.nature.com/articles/s41587-021-01146-5
            PLL(x) = \sum_{i=1}^{L} \log{p(x_i|x_{-i})})

    Args:
        model (ESMEncoder): An ESMEncoder model. (In the future any masked language model can be used)
        save_dir (Path, optional): Directory to save query results to or load from. Defaults to DATA_PATH.
    """

    def __init__(self, model, save_dir: Path = DATA_PATH):
        self.model = model
        self.save_dir = save_dir

        if not isinstance(model, ESMEncoder):
            raise NotImplementedError(
                "Only ESMEncoder is supported for now, "
                "will need to implement a more general aa_idxs mapping for other models."
            )
        self._output_to_aa_idx = torch.tensor(
            [model.alphabet.all_toks.index(i) for i in NATURAL_AA],
            dtype=torch.long,
        )
        self._idx_to_aa = {val: i for i, val in enumerate(NATURAL_AA)}

    def _get_masked_token_probs(self, seqs: str, masked_pos: List[List[int]]) -> torch.Tensor:
        """
        Get the probabilities of the masked tokens in the given sequences.

        Args:
            seqs (str): A list of protein sequences to evaluate.
            masked_pos (List[List[int]]): A list of masked positions for each sequence. These
                positions must coincide with the masked positions in `seqs`.

        Returns:
            torch.Tensor: A tensor of probabilities with shape [masked_pos, vocab_size].
        """
        token_probs = self.model(seqs, output="probabilities", drop_padding=True, to_numpy=False)
        aa_probs = [p[pos, :] for p, pos in zip(token_probs, masked_pos)]
        return aa_probs  # [masked_pos, vocab_size]

    @staticmethod
    def _construct_queries(variants: List[Variant], wt_seq: str) -> List[Tuple[str, List[int]]]:
        """
        Construct queries for the mutated and wildtype sequences where they are different.
        This is useful because many queries will be the same, so we can save computation by
        only computing the probabilities for each unique query.

        Args:
            variants (List[Variant]): List of variant objects.
            wt_seq (str): Wildtype sequence.

        Returns:
            List[Tuple[str, List[int]]]: List of queries as tuples (masked_query_seq, masked_pos).
        """
        queries = []
        for variant in variants:
            variant_seq = variant.get_sequence(wt_seq)
            for mutation in variant.mutations:
                for seq in [wt_seq, variant_seq]:
                    # Mask the sequence at the mutation position
                    masked_pos = mutation.pos
                    masked_seq = apply_mask(seq, masked_pos)
                    # Add the query to the list
                    queries.append((masked_seq, masked_pos))
        _n_naive_queries = len(queries)
        logger.info(f"Constructed {len(queries):,} naive queries. Deduplicating...")

        queries = list(set(queries))
        _n_deduplicated_queries = len(queries)

        logger.info(
            f"Removed duplicates ({100 - 100.*(_n_deduplicated_queries / _n_naive_queries):.1f}%). Remaining: {_n_deduplicated_queries:,}"
        )

        return queries

    def _perform_queries(self, queries: List[Tuple[str, List[int]]], batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Perform masked language model queries for the given list of queries.

        Args:
            queries (List[Tuple[str, List[int]]]): A list of queries as tuples (masked_query_seq, masked_pos).
            batch_size (int, optional): The batch size for processing queries. Defaults to 8.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of query results with masked query sequences as keys and
                log probability tensors as values.
        """
        query_results = {}

        logger.info("Performing queries...")
        for query_batch in tqdm(to_batch(queries, batch_size=batch_size)):
            seq_batch, pos_batch = zip(*query_batch)
            probs = self._get_masked_token_probs(seq_batch, pos_batch)
            for seq, prob in zip(seq_batch, probs):
                query_results[seq] = prob.cpu().log()[self._output_to_aa_idx]

        return query_results

    def _load_query_results(self, query_results_path) -> Dict[str, torch.Tensor]:
        with open(query_results_path, "rb") as f:
            query_results = dill.load(f)
        return query_results

    def _save_query_results(self, query_results: Dict[str, torch.Tensor], query_results_path):
        with open(query_results_path, "wb") as f:
            dill.dump(query_results, f)

    def _get_fitness_from_query_results(
        self, variant: Variant, wt_seq: str, query_results: Dict[str, torch.Tensor]
    ) -> float:
        """
        Calculate the fitness of a protein variant from the query results.

        Args:
            variant (Variant): A Variant object representing the protein variant.
            wt_seq (str): The wildtype sequence.
            query_results (Dict[str, torch.Tensor]): A dictionary of query results with masked query sequences
                as keys and log probability tensors as values.

        Returns:
            float: The fitness score of the protein variant.
        """
        fitness = 0.0
        for mutation in variant.mutations:
            wt_seq_masked = apply_mask(wt_seq, mutation.pos)
            mut_seq_masked = apply_mask(variant.get_sequence(wt_seq), mutation.pos)
            wt_aa = self._idx_to_aa[mutation.from_seq]
            mut_aa = self._idx_to_aa[mutation.to_seq]
            fitness += (query_results[mut_seq_masked][mut_aa] - query_results[wt_seq_masked][wt_aa]).item()
        return fitness

    def evaluate(self, dataset, batch_size: int = 8) -> Sequence[float]:
        """
        Evaluate a dataset of protein variants.

        Args:
            dataset (Dataset): A Dataset object containing protein variants and wildtype sequences.

        Returns:
            Sequence[float]: A list of fitness scores, one for each protein variant.
        """
        query_save_path = self.save_dir / f"{dataset.name}-{self.model.name}-query_results.dill"

        # Check if query results have already been saved
        if query_save_path.exists():
            query_results = self._load_query_results(query_save_path)
        else:
            logger.info(f"Query results not found at `{query_save_path}` . Computing...")
            variants = dataset.data.variant.apply(Variant.from_str)
            queries = self._construct_queries(variants, dataset.wildtype_seq)
            query_results = self._perform_queries(queries, batch_size=batch_size)
            self._save_query_results(query_results, query_save_path)

        # Calculate fitness scores for each protein variant
        fitness_scores = dataset.data.variant.apply(Variant.from_str).apply(
            self._get_fitness_from_query_results,
            wt_seq=dataset.wildtype_seq,
            query_results=query_results,
        )

        return fitness_scores


def get_sequence_probabilities_without_masking(
    model: ESMEncoder,
    seqs: Sequence[str],  # TODO: Generalise to any model
) -> torch.Tensor:
    aa_probs = model(seqs, output="probabilities", drop_padding=True, to_numpy=False)
    return aa_probs


def get_sequence_probabilities_with_masking(
    model: ESMEncoder,  # TODO: Generalise to any model
    seq: str,
    batch_size: int = 16,
    mask_token: str = "<mask>",
) -> torch.Tensor:
    # NOTE: This is slow because it requires len(seq) forward passes through
    #  the model, as we have to mask out each token in the sequence individually.
    seqs = [apply_mask(seq, i, mask_token=mask_token) for i in range(len(seq))]
    token_probs = []
    for seq in tqdm(to_batch(seqs, batch_size)):
        token_probs.append(model(seq, output="probabilities", drop_padding=True, to_numpy=False))
    token_probs = torch.concat(token_probs, dim=0)  # [seq_len, seq_len, alphabet_len]
    # Select only the probabilities of the masked token (ie. the diagonal)
    token_probs = torch.einsum("iik->ik", token_probs)  # [seq_len, alphabet_len]
    return token_probs


def pseudo_log_likelihood(p: torch.Tensor) -> float:
    # NOTE: Pseudo-log-likelihood (PLL) as defined in
    # https://www.nature.com/articles/s41587-021-01146-5
    #   PLL(x) = \sum_{i=1}^{L} \log{p(x_i|x_{-i})})
    # p: [seq_len]
    assert len(p.shape) == 1  # TODO: Generalise to batches
    return torch.sum(torch.log(p))


def pseudo_perplexity(p: torch.Tensor) -> float:
    # NOTE: Pseudo-perplexity (PPL) as defined in
    # https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1
    #   PPL(x) = \exp( -\frac{1}{L} \sum_{i=1}^{L} \log{p(x_i|x_{-i})})
    #          = \exp( -\frac{1}{L} PLL(x))
    # p: [seq_len]
    assert len(p.shape) == 1  # TODO: Generalise to batches
    return torch.exp(-torch.sum(torch.log(p)) / len(p))


def calculate_pseudo_perplexity(
    model: ESMEncoder,  # TODO: Generalise to any model
    seq: Sequence[str],
    batch_size: int = 16,
    mask_token: str = "<mask>",
) -> float:
    masked_probs_all = get_sequence_probabilities_with_masking(
        model, seq, batch_size, mask_token
    )  # [seq_len, vocab_size]

    # Select diagonal along sequence axis and vocab token according to input sequence
    # TODO: Fix selection [0, 1:-1, :] for more general padding
    seq_one_hot = model.one_hot(seq)[0, 1:-1, :]  # [seq_len, vocab_size]
    masked_probs_target_token = masked_probs_all[seq_one_hot == 1]  # [seq_len]
    return pseudo_perplexity(masked_probs_target_token)
