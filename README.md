train_pretrain = 专家损失 + NTP交叉熵损失
train_full_sft = 专家损失 + NTP交叉熵损失
train_lora     = 专家损失 + NTP交叉熵损失

train_dpo      = 专家损失 + DPO损失 -logσ(βlog Πw/Rw - βlog Πl/Rl) = -logσ(βlog Πw/Πl - βlog Rw/Rl)
train_grpo
train_reason

train_ppo
train_spo
train_distillation