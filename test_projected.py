import re
import torch
import torch.nn as nn


def build_projected():
    projector_type = "mlp2x_gelu"
    mm_hidden_size = 1024
    hidden_size = 4096

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

def load_projected_params(projected_model, checkpoint):
    for i, (name, module) in enumerate(projected_model.named_modules()):
        # 判断模块是否是 Linear 层
        if isinstance(module, nn.Linear):
            layer_index = name
            # 动态加载与该层对应的权重和偏置
            weight_name = f"model.mm_projector.{layer_index}.weight"
            bias_name = f"model.mm_projector.{layer_index}.bias"

            if weight_name in checkpoint and bias_name in checkpoint:
                module.weight.data = checkpoint[weight_name]
                module.bias.data = checkpoint[bias_name]
                print(f"Loaded weights and biases for {name}")

    return projected_model

if __name__ == '__main__':
    projected_model = build_projected()
    print(projected_model)

    model_path = "/data/gongoubo/VQA/LLaVA/model_hub/llava-v1.5-7b/mm_projector.bin"

    projected_model_checkpoint = torch.load(model_path)
    # for name, value in projected_model_checkpoint.items():
    #     print(name)

    projected_model = load_projected_params(projected_model, projected_model_checkpoint)

    # torch.float16必须在gpu上进行计算
    projected_model = projected_model.to("cuda:0")
    input = torch.randn((1, 512, 1024), dtype=torch.float16).to("cuda:0")

    output = projected_model(input)
    print(output.shape)