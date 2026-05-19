import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = load_dataset("json", data_files=data_path, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = self.tokenizer(
            sample["text"],
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        tokens = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
