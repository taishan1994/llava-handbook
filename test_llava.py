# 这个文件时真正用于测试llava

# 读取llava的tokenizer、config以及模型权重，后续用于加载成我们自定义的
# =================================
"""
LLAVA的核心推理代码是：
from llava.model.builder import load_pretrained_model
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
"""
# =================================
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# 对比llava里面模型的权重以及transformers里面的AutoModelForCausalLM权重看看有什么差异
from llava.model.builder import load_pretrained_model

# model_path = "/data/gongoubo/VQA/LLaVA/model_hub/llava-v1.5-7b"
model_path = "/data/gongoubo/VQA/LLaVA/checkpoints/llava-v1.5-7b-merge"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, "llava", device="cpu")

# 打印llava模型的权重

state_dict = {}

for name, module in model.named_parameters():
    # print(name, module.shape)
    if "vision_tower" not in name and "projector" not in name:
        state_dict[name] = module.cpu()

# 实际上llava里面的权重分为三个部分：language_model、vision_model、projected_model
# language_model即语言模型，也就是文本编码器，vision_model为视觉模型，也就是视觉编码器，projected_model则是将视觉编码器输出的向量维度映射到语言编码器输入的维度。

print("=" * 100)

# j接下来我们先取出llava里面语言编码器的权重并用于进行语言的生成
llama_model_path = "/data/gongoubo/VQA/LLaVA/model_hub/shakechen/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(llama_model_path)


init_llama_model = AutoModelForCausalLM.from_config(config)
init_llama_model.load_state_dict(state_dict)
init_llama_model.half().cuda().eval()

# conversation = [{"role": "user", "content": "who are you?"}]

# inps = llama_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

inps = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: 用中文回答你是谁？ ASSISTANT:"

print(inps)

inps = llama_tokenizer(inps, add_special_tokens=False, return_tensors="pt")
print(inps)


output = init_llama_model.generate(
    input_ids=inps.input_ids.to(init_llama_model.device),
    attention_mask=inps.attention_mask.to(init_llama_model.device),
    max_new_tokens = 512,
)
print(output)
print(llama_tokenizer.decode(output[0], skip_special_tokens=False))


# 下面的是结合图片的样例
# 输入的是图像以及文本
input = {
            "conversations": [
                {"from": "human", "value": "<image>\n请用中文描述上述图片。"},
                {"from": "gpt", "value": "【二手9成新】作家德富曼诺夫传略"}
            ],
            "id": "00514175-0237",
            "image": "/data/gongoubo/VQA/LLaVA/serve_images/2024-08-23/b939abf2c4553ce07e642170aee3a3d7.jpg"}

special_tokens = {
        'additional_special_tokens': ['<image>']  # 可以添加多个特殊符号
    }
llama_tokenizer.add_special_tokens(special_tokens)

# 转换为input_ids
input_ids = []
labels = []
sources = input["conversations"]
# print(sources)
system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
for i, source in enumerate(sources):
    value = source["value"]
    if i == 0:
        value = f"{system} USER: {value}"
    if i % 2 == 0:
        _input = "<s>" + f"{value} ASSISTANT:"
        _input_ids = llama_tokenizer(_input, add_special_tokens=False)["input_ids"]

        input_ids += _input_ids

print(input_ids)
print(llama_tokenizer.decode(input_ids, skip_special_token=True))

# 为文本生成embedding
import torch
input_embeddings = init_llama_model.get_input_embeddings()
# generated_sequence_embedding = input_embeddings(input_ids).half().to(model.device)
splits = []
image_token_ids = input_ids.index(32000)

print(image_token_ids)

input_embedding_part1 = input_embeddings(torch.tensor([input_ids[:image_token_ids]]).to(init_llama_model.device))
input_embedding_part2 = input_embeddings(torch.tensor([input_ids[image_token_ids+1:]]).to(init_llama_model.device))

print(input_embedding_part1.shape)
print(input_embedding_part2.shape)

import os
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 32000


def encode_image(images, vision_tower=None, projected_model=None):
    output = vision_tower(images, output_hidden_states=True)
    output = output.hidden_states[-2]
    print("clip倒数第二层输出：", output.shape)
    output = projected_model(output)
    print("project输出：", output.shape)
    return output


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
                module.weight.data = checkpoint[weight_name].to(torch.float32)
                module.bias.data = checkpoint[bias_name].to(torch.float32)
                print(f"Loaded weights and biases for {name}")

    return projected_model

from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel

projected_model = build_projected()
print(projected_model)

model_path = "/data/gongoubo/VQA/LLaVA/model_hub/llava-v1.5-7b/mm_projector.bin"

projected_model_checkpoint = torch.load(model_path)
# for name, value in projected_model_checkpoint.items():
#     print(name)
projected_model = load_projected_params(projected_model, projected_model_checkpoint)



if llama_tokenizer.pad_token_id is None:
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336",
                                        cache_dir="/data/gongoubo/VQA/LLaVA//model_hub/",
                                               torch_dtype=torch.float32)
# print(vision_tower)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="/data/gongoubo/VQA/LLaVA//model_hub/")
image = "/data/gongoubo/VQA/LLaVA/serve_images/2024-08-23/b939abf2c4553ce07e642170aee3a3d7.jpg"
image = Image.open(image).convert('RGB')

image = processor(images=image, return_tensors='pt')['pixel_values'][0]
image = image.unsqueeze(0)
print(image.shape)
image_embedding = encode_image(image, vision_tower=vision_tower, projected_model=projected_model)

image_embedding = image_embedding.half().to(init_llama_model.device)
inp_embedding = torch.cat((input_embedding_part1, image_embedding, input_embedding_part2), dim=1)

print(image_embedding.shape)

output = init_llama_model.generate(inputs_embeds=inp_embedding,  max_new_tokens = 2048)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=False)
print(decoded_output)
