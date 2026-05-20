import os
import sys

__package__ = "trainer"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.llm_dataset import SFTDataset
from trainer_utils import init_model

model, tokenizer = init_model()
dataset = SFTDataset("../dataset/sft_mini_512.jsonl", tokenizer, max_length=512)
for name, module in model.named_modules():
    print(name, module)
