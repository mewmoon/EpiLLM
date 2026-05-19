import os
import sys

__package__ = "trainer"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from transformers import AutoTokenizer
from model.model import EpiForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_model(
    lm_config=None,
    from_weight="none",
    tokenizer_path="../tokenizer",
    save_dir="../out",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EpiForCausalLM(lm_config)
    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = (
            f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        )
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    return model.to(device), tokenizer
