import swanlab
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 1. 初始化 SwanLab：在这里定义你的超参数，它会自动存入系统
# project 是项目名，experiment_name 是本次运行的唯一标识
swanlab.init(
    project="rag-demo",
    experiment_name="test",
    config={"learning_rate": 0.001, "epochs": 10, "batch_size": 16},
)

# 模拟一个简单的模型
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 模拟训练过程
global_step = 0
for epoch in range(10):
    for batch in range(50):
        global_step += 1
        # 模拟生成数据
        inputs = torch.randn(16, 10)
        targets = torch.randn(16, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 2. 记录指标：每一轮都 log 一下 loss
        # swanlab.log 会自动帮你把数据汇总成精美的图表
        # 一次性记录 loss、学习率、以及模型输出的分布情况
        swanlab.log(
            {
                "train/loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "metrics/accuracy": 0.85,  # 如果有验证集指标
                "step": global_step,
            }
        )

    print(f"Epoch {epoch} finished.")
