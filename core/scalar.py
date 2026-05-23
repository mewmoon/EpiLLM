import torch
import torch.nn as nn
import torch.optim as optim
import swanlab

swanlab.init(
    project="scale-demo",
    experiment_name="amp-training-run-scaled",
    config={"learning_rate": 0.01, "epochs": 5, "batch_size": 32},
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(
    nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1)
).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

scaler = torch.cuda.amp.GradScaler()


def get_data():
    for _ in range(500):
        yield torch.randn(32, 10).to(device), torch.randn(32, 1).to(device)


global_step = 0
for epoch in range(5):
    for inputs, targets in get_data():
        global_step += 1
        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # 缩放并反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪：这是工业界标准，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        swanlab.log(
            {
                "train/loss": loss.item(),
                "scale_factor": scaler.get_scale(),
            },
            step=global_step,
        )

    print(f"Epoch {epoch} finished.")
