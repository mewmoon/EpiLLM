import os
import sys

__package__ = "trainer"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.llm_dataset import *
from trainer_utils import init_model
from torch.utils.data import Subset

model, tokenizer = init_model()
dataset = RLAIFDataset("../dataset/rlaif.jsonl", tokenizer, max_length=512)
print(dataset[0])
