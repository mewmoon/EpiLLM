import os
import sys

__package__ = "trainer"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from model.model_lora import apply_lora, freeze_model, save_lora
from trainer.trainer_utils import init_model

model, _ = init_model()
# model = torch.compile(model)
apply_lora(model, rank=8)
print(model)


lora_params, _, _ = freeze_model(model)
# print(lora_params)
# lora_params传入优化器
lora_path = "../out/lora.pth"
# save_lora(model, lora_path)

print(model.model.blocks[0].attn.q_proj.lora.A.weight.data)
model.load_state_dict(torch.load(lora_path), strict=False)  # sft的strict =True
print(model.model.blocks[0].attn.q_proj.lora.A.weight.data)
