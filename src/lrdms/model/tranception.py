import json
import os
import pathlib

import pandas as pd
import torch
import tranception
import tranception.config
import tranception.model_pytorch
from transformers import PreTrainedTokenizerFast

from src.constants import PROJECT_PATH
from src.utils.logutils import get_logger

# TODO: Tranception is not pip-installable yet. Fix this by adding scripts to run as it cannot be added as easy dependency.
logger = get_logger(__file__)
pathlib.Path(os.environ.get("MODELS_PATH", PROJECT_PATH))
TRANCEPTION_REPO_DIR = PROJECT_PATH / "Tranception"
TRANCEPTION_MODEL_DIR = pathlib.Path(os.environ.get("MODELS_PATH", PROJECT_PATH)) / "Tranception_Large"


def _load_tranception_base_config(
    repo_dir=TRANCEPTION_REPO_DIR, model_dir=TRANCEPTION_MODEL_DIR
) -> tranception.config.TranceptionConfig:
    assert repo_dir.exists() and repo_dir.is_dir(), f"Repo dir {repo_dir} does not exist or is not a directory"
    assert model_dir.exists() and model_dir.is_dir(), f"Model dir {model_dir} does not exist or is not a directory"

    with open(model_dir / "config.json") as f:
        base_cfg = tranception.config.TranceptionConfig(**json.load(f))

    base_cfg.tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(repo_dir / "tranception/utils/tokenizers/Basic_tokenizer"),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    return base_cfg


def run_tranception(
    dataset,
    msa_file: str = None,
    device: str = "cuda",
    scoring_mirror: bool = True,
    batch_size_inference: int = 100,
    repo_dir: pathlib.Path = TRANCEPTION_REPO_DIR,
    model_dir: pathlib.Path = TRANCEPTION_MODEL_DIR,
) -> pd.DataFrame:
    wt = dataset.wildtype_seq

    # Set up & populate tranception model configuration
    logger.info("Load tranception config ...")
    cfg = _load_tranception_base_config(repo_dir, model_dir)

    cfg.full_protein_length = len(wt)
    if msa_file is not None:
        cfg.retrieval_aggregation_mode = "aggregate_substitution"
        cfg.MSA_filename = msa_file
        cfg.MSA_start = 0
        cfg.MSA_end = len(wt)

    # Load pre-trained tranception models and configure for this dataset
    logger.info("Loading pretrained model from %s ..." % (model_dir / "pytorch_model.bin"))
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(
        model_dir / "pytorch_model.bin", config=cfg
    )

    # Prepare the datsaset for use with Tranception
    logger.info("Preparing dataset for inference ...")
    df_for_tranception = dataset.data[["variant", "sequence"]].rename(
        columns={"sequence": "mutated_sequence", "variant": "mutant"}
    )
    df_for_tranception["mutant"] = df_for_tranception["mutant"].str.replace(",", ":")
    df_for_tranception = df_for_tranception[df_for_tranception.mutant.str.len() > 1]

    logger.info("Performing inference ...")
    with torch.inference_mode():
        model.eval()
        model.to(device)
        result = model.score_mutants(
            df_for_tranception,
            target_seq=wt,
            scoring_mirror=scoring_mirror,
            batch_size_inference=batch_size_inference,
        )

    logger.info("Post-processing dataset ...")
    result = pd.merge(result, dataset.data, left_on="mutated_sequence", right_on="sequence")
    result = result[["variant", "avg_score_L_to_R", "avg_score_R_to_L", "avg_score", "fitness"]]

    return result
