import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(0, 0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    layers_to_replace = []
    for name, module in model.named_modules():
        # 方阵加上module.weight.shape[0] == module.weight.shape[1]
        if isinstance(module, nn.Linear):
            layers_to_replace.append((name, module))

    for name, module in layers_to_replace:
        # device = next(model.parameters()).device
        lora = LoRA(module.weight.shape[1], module.weight.shape[0], rank).to(
            model.device
        )
        setattr(module, "lora", lora)
        old_forward = module.forward

        def forward_with_lora(x, layer1=old_forward, layer2=lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora


def freeze_model(model):
    lora_params = []
    total_params = trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora" in name:
            param.requires_grad = True
            lora_params.append(param)
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(f"总参数{total_params/ 1e6:.3f} M,可训练参数{trainable_params/ 1e6:.3f} M")
    return lora_params, total_params, trainable_params


def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)  # 兼容compile
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name  # 兼容DDP
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
    return state_dict
