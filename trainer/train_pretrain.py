from transformers import AutoTokenizer
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dataset.llm_dataset import PretrainDataset
from model.model import EpiForCausalLM


def init_model(
    lm_config=None,
    from_weight="none",
    tokenizer_path="../tokenizer",
    save_dir="../out",
    device="cuda",
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


model, tokenizer = init_model()
dataset = PretrainDataset("../dataset/pretrain.jsonl", tokenizer, max_length=512)

input_ids = dataset[0][0].unsqueeze(0).to("cuda")
labels = dataset[0][1].unsqueeze(0).to("cuda")
outputs = model(input_ids=input_ids, labels=labels)
print(outputs.loss)
