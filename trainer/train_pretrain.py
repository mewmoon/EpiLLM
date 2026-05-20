import os
import sys

__package__ = "trainer"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset.llm_dataset import PretrainDataset, SFTDataset
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
NUM_EPOCHS = 1  # 根据需求修改圈数
# ====================================================

# 2. 初始化模型和数据
model, tokenizer = init_model()
model.to(DEVICE)
apply_lora(model)
lora_params, _, _ = freeze_model(model)

# dataset = PretrainDataset("../dataset/pretrain.jsonl", tokenizer, max_length=512)
dataset = SFTDataset("../dataset/sft_mini_512.jsonl", tokenizer, max_length=512)


train_loader = DataLoader(
    dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, drop_last=True
)

# 3. 设置优化器与混合精度加速
# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE, weight_decay=0.1)
scaler = GradScaler()  # 混合精度 AMP 缩放器，防止 FP16 梯度下溢


# 4. 定义通用的训练 Epoch 函数
def train_epoch(epoch, model, dataloader, optimizer, scaler, device, save_dir):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    # 真实记录更新了多少个真正的“大 Batch”
    global_step = 0

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 开启混合精度（AMP）前向传播，大幅节省显存
        with autocast():
            outputs = model(input_ids=input_ids, labels=labels)
            # 兼容处理返回格式
            loss = outputs.loss if hasattr(outputs, "loss") else outputs
            # 关键：损失值需要除以累积步数，进行平均
            loss = loss / GRAD_ACCUM_STEPS

        # 反向传播（缩放梯度）
        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM_STEPS

        # 当达到指定的累积步数，或者已经到了最后一个 batch 时，更新权重
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(
            dataloader
        ):
            global_step += 1

            # 梯度裁剪，防止大模型训练梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 步进优化器并清空梯度
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 计算当前大 Batch 的平均真实 Loss 并打印
            current_loss = total_loss / (batch_idx + 1)
            print(
                f"Epoch: {epoch} | Step: {global_step} (Batch_End: {batch_idx+1}) | Loss: {current_loss:.4f}"
            )

            # 核心需求：每过 10 个逻辑大 Batch（相当于跑了 10 * 16 = 160 个小 batch）存一次临时 Checkpoint
            if global_step % 10 == 0:
                ckpt_path = os.path.join(
                    save_dir, f"checkpoint_epoch{epoch}_step{global_step}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)
                print(f"--> Saved checkpoint to {ckpt_path}")

    avg_loss = total_loss / len(dataloader)
    print(f"=== Epoch {epoch} 训练完成 | 整个数据集平均 Loss: {avg_loss:.4f} ===")
    return avg_loss


# 5. 开始执行全流程训练
if __name__ == "__main__":
    print(
        f"训练启动... 目标总 Batch_Size: {TARGET_BATCH_SIZE} (物理 Batch: {MICRO_BATCH_SIZE} * 梯度累积: {GRAD_ACCUM_STEPS})"
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = train_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
            save_dir=SAVE_DIR,
        )

    # 核心需求：最终结果放到 out/ 文件夹下
    final_model_path = os.path.join(SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"🎉 训练结束！最终模型成功保存至: {final_model_path}")
