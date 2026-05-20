import os
import sys

from altair import sample

__package__ = "trainer"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.llm_dataset import *
from trainer_utils import init_model
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# ==================== 配置超参数 ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "../out"
os.makedirs(SAVE_DIR, exist_ok=True)
TARGET_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = TARGET_BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

# 策略模型
model, tokenizer = init_model()

# 参考模型
ref_model, _ = init_model()
ref_model.eval()
ref_model.requires_grad_(False)

dataset = DPODataset("../dataset/dpo.jsonl", tokenizer, max_length=512)
# indices = range(1000)
# dataset = Subset(dataset, indices)
train_loader = DataLoader(
    dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, drop_last=True
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
scaler = GradScaler()


def logits_to_log_probs(logits, labels):
    # logits: batch_size,seq_len,vocab_size
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)
    # logits: batch_size,seq_len
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1):

    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止零长度mask
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    # ref_log_probs: batch_size
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, beta=0.1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    global_step = 0

    for step, batch in enumerate(loader):
        x_chosen = batch["x_chosen"].to(device)
        x_rejected = batch["x_rejected"].to(device)
        y_chosen = batch["y_chosen"].to(device)
        y_rejected = batch["y_rejected"].to(device)
        mask_chosen = batch["mask_chosen"].to(device)
        mask_rejected = batch["mask_rejected"].to(device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 开启混合精度（AMP）前向传播，大幅节省显存
        with autocast():
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / GRAD_ACCUM_STEPS

        # 反向传播（缩放梯度）
        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM_STEPS

        # 当达到指定的累积步数，或者已经到了最后一个 batch 时，更新权重
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            global_step += 1

            # 梯度裁剪，防止大模型训练梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 步进优化器并清空梯度
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 计算当前大 Batch 的平均真实 Loss 并打印
            current_loss = total_loss / (step + 1)
            print(
                f"Epoch: {epoch} | Step: {global_step} (Batch_End: {step+1}) | Loss: {current_loss:.4f}"
            )

            # 核心需求：每过 10 个逻辑大 Batch（相当于跑了 10 * 16 = 160 个小 batch）存一次临时 Checkpoint
            if global_step % 10 == 0:
                ckpt_path = os.path.join(
                    SAVE_DIR, f"checkpoint_epoch{epoch}_step{global_step}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)
                print(f"--> Saved checkpoint to {ckpt_path}")

    avg_loss = total_loss / len(loader)
    print(f"=== Epoch {epoch} 训练完成 | 整个数据集平均 Loss: {avg_loss:.4f} ===")
    return avg_loss


if __name__ == "__main__":
    print(
        f"训练启动... 目标总 Batch_Size: {TARGET_BATCH_SIZE} (物理 Batch: {MICRO_BATCH_SIZE} * 梯度累积: {GRAD_ACCUM_STEPS})"
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = train_epoch(epoch=epoch, loader=train_loader)

    # 核心需求：最终结果放到 out/ 文件夹下
    final_model_path = os.path.join(SAVE_DIR, "dpo_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"🎉 训练结束！最终模型成功保存至: {final_model_path}")
