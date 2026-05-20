import os
import sys

__package__ = "trainer"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from dataset.llm_dataset import *
from trainer_utils import init_model
from model.model_lora import apply_lora, freeze_model

# ==================== 配置超参数 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "../out"
os.makedirs(SAVE_DIR, exist_ok=True)

# 显存优化：目标大 Batch 是 64。我们设微 Batch 是 4，累积步数就是 16 (4 * 16 = 64)
TARGET_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = TARGET_BATCH_SIZE // MICRO_BATCH_SIZE

LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

num_generations = 3
# ======================================================


# Policy模型
model, tokenizer = init_model()
# Reference模型
ref_model, _ = init_model()
ref_model = ref_model.eval().requires_grad_(False)
# Reward模型
reward_model_path = "gpt2"
reward_model = AutoModel.from_pretrained(
    reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
)
reward_model = reward_model.to(DEVICE).eval().requires_grad_(False)
reward_tokenizer = AutoTokenizer.from_pretrained(
    reward_model_path, trust_remote_code=True
)


dataset = RLAIFDataset("../dataset/rlaif.jsonl", tokenizer, max_length=512)
train_loader = DataLoader(
    dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, drop_last=True
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
scaler = GradScaler()


def grpo_train_epoch(
    epoch,
    loader,
    iters,
    ref_model,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    pass


if __name__ == "__main__":
    print(
        f"训练启动... 目标总 Batch_Size: {TARGET_BATCH_SIZE} (物理 Batch: {MICRO_BATCH_SIZE} * 梯度累积: {GRAD_ACCUM_STEPS})"
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = grpo_train_epoch(
            epoch=epoch,
            dataloader=train_loader,
            ref_model=ref_model,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
        )

    # 核心需求：最终结果放到 out/ 文件夹下
    final_model_path = os.path.join(SAVE_DIR, "grpo_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"🎉 训练结束！最终模型成功保存至: {final_model_path}")
