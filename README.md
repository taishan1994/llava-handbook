# llava-handbook
对llava官方代码的一些学习笔记

# 可视化部署

```shell
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

conda deactivate llava

权重这里下载：https://huggingface.co/liuhaotian/llava-v1.5-7b
sh run_controller.sh
sh run_worker.sh # 这里面要修改llava-v1.5-7b的路径
sh run_gradio,sh
```

# 推理代码解析

- llava/conversation.py：用于存放Prompt模板
- llava/model/builder.py：用于加载权重、tokenizer、图像处理器以及上下文长度
- llava/model/language_model/llava_llama.py：LlavaLlamaForCausalLM

llava/model/builder.py中要注意：

1. 实际上使用的tokenizer还是语言模型的。
2. vision_tower可以在这里重新加载参数，并且使用的图像处理是vision_tower的image_processor。

llava/model/language_model/llava_llama.py中要注意：

1. LlavaLlamaForCausalLM继承了transformers里面的LlamaForCausalLM以及llava/model/llava_arch.py里面的LlavaMetaForCausalLM。需要重写get_model方法。在LlavaMetaForCausalLM中__init__的时候定义了self.model = LlavaLlamaModel(config)，而LlavaLlamaModel继承了transformers里面的LlamaModel以及llava/model/llava_arch.py里面的LlavaMetaModel。LlavaMetaModel里面__init__才是真正定义了vision_tower以及vision_projector。到这里模型的定义就基本完成了。LlavaMetaModel还定义了一个class_config=LlavaConfig，LlavaConfig继承了LlamaConfig，主要修改了model_type属性。llava_llama.py的末尾需要将定义的config和model注册到transformers中。
2. llava/model/multimodal_encoder/builder.py里面定义了加载的vision_tower，根据config.json里面的mm_vision_tower以及vision_tower_cfg进行配置，主要是使用到了llava/model/multimodal_encoder/clip_encoder.py下的CLIPVisionTower。
3. llava/model/multimodal_projector/builder.py里面定义了加载的projector。

llava/model/multimodal_encoder/clip_encoder.py中要注意：

   1.从transformers里面导入了CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig，实际上使用的就是这些，同时在加载的时候会把梯度设置为False。在模型得到输出后还有一个feature_select函数用于选择特征。

llava/mm_utils.py里面要注意：

1. 里面定义了对图像的处理，process_images, load_image_from_base64, tokenizer_image_token。
2. process_images中需要传入images，image_processor，model_config。在config.json中image_aspect_ratio设置为pad，然后调用expand2square方法以及image_processor的preprocess方法。我们先去研究下image_processor是个什么东西。主要就是对图片进行一些操作然后输入到视觉编码器里面，需要注意的是要注意输入和输出。假设原始图像大小为 224x224x3。分割为 16x16 的图像块，这意味着每个小块的尺寸为 16x16x3。224x224 的图像总共可以分割为 (224 / 16) x (224 / 16) = 14 x 14 = 196 个图像块。每个图像块展平成一个长度为 16 * 16 * 3 = 768 的向量。

# 训练代码解析

llava训练需要经过两个阶段：

1. 第一阶段：冻结住文本编码器和图片编码器，只训练对图像的映射权重。
2. 第二阶段：冻结住图像编码器，训练文本和图像的映射权重。 注意到，llava计算损失除了assistant的回答+eos_token外，还会计算human的eos_token。

## 使用Lora进行训练

### 数据预处理

数据位于：playground/data下，来源：https://wukong-dataset.github.io/wukong-dataset/download.html 使用的是Wukong-Test数据。使用/convert_wukong.py中代码将悟空数据集转换为所需要的训练格式，

```python
import pandas as pd
import json

data = pd.read_csv("playground/data/wukong_test.csv")
print(data.columns)
res = []
for d in data.iterrows():
    d = d[1]
    text = d["text"]
    image = d["dir"]
    tmp = {
        "conversations": [{"from":"human", "value":"<image>\n为该图片生成标注。"},
                          {"from":"gpt", "value":text}],
        "id": image.split(".")[0],
        "image": "wukong_test_images/{}".format(image)
    }
    res.append(tmp)

with open("playground/data/wukong_test.jsonl", "w") as fp:
    fp.write(json.dumps(res, ensure_ascii=False))
```

需要注意：

- tmp为训练数据需要的格式。其中conversations根据不同的模型可能不大一样。image是实际的图片所在的位置，也在playground/data/下。id是每个图片的唯一标识。

### 训练

训练脚本位于script/v1_5/fintune_lora_wukong.sh。在LLaVA目录下执行：

```shell
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/gongoubo/VQA/LLaVA/model_hub/llava-v1.5-7b/ \
    --version v1 \
    --data_path ./playground/data/wukong_test.jsonl \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/gongoubo/VQA/LLaVA/model_hub/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
```

训练完成后，使用scripts/merge_lora_wukong.py将权重合并到模型中：

```python
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(model_path, model_base, save_model_path):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device_map='cpu')

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)


if __name__ == "__main__":

    model_path = "/data/gongoubo/VQA/LLaVA/checkpoints/llava-v1.5-7b-lora/"
    model_base = "/data/gongoubo/VQA/LLaVA/model_hub/llava-v1.5-7b/"
    save_model_path = "/data/gongoubo/VQA/LLaVA/checkpoints/llava-v1.5-7b-merge/"
    merge_lora(model_path, model_base, save_model_path)
```

最后按照之前所讲的，启动worker即可：

```shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://0.0.0.0:10000 --port 40000 --worker http://0.0.0.0:40000 --model-path /data/gongoubo/VQA/LLaVA/checkpoints/llava-v1.5-7b-merge
```

还需要重点关注llava/model/llava_arch.py的prepare_inputs_labels_for_multimodal函数，这里是对数据进行预处理操作，需要理解里面每一个步骤的作用。

# 常见错误

- ImportError: cannot import name 'LlavaLlamaForCausalLM' from 'llava.model' ：https://github.com/haotian-liu/LLaVA/issues/1101
- size mismatch for 0.weight：将zero3.json修改为zero2.json

# 代码学习

## test_clip.py

主要是使用clip作图像编码器，将图片编码成特征。

```python
import numpy as np
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPProcessor


# 可参考：https://colab.research.google.com/drive/1pPHiwHUnM3zmTLtMcNL2zTu2M6FdKtRK?usp=sharing
# ====================
# 加载模型和图像处理器
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="/data/gongoubo/VQA/LLaVA//model_hub/")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
print(processor)
# ====================

# ====================
# 读取图片
img_path = "/data/gongoubo/VQA/LLaVA/llava/serve/examples/waterview.jpg"
from PIL import Image
image = Image.open(img_path).convert('RGB')
print(np.array(image).shape)
image = processor(images=image, return_tensors='pt')

print(image["pixel_values"].shape)

output = model(image["pixel_values"], output_hidden_states=True)
print(len(output.hidden_states))
print(output.hidden_states[-1].shape)
```

## test_projected.py

主要是根据配置文件定义了一个projector，主要是将图像特征的最后一维和文本特征的最后一维进行对齐：

```python
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
```

## test_llama2.py

主要是测试语言模型的输入和输出。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/data/gongoubo/VQA/LLaVA/model_hub/shakechen/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16)

device = "cuda:0"
model = model.to(device).eval()

conversation = [{"role": "user", "content": "who are you"}]

inps = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

print(inps)

inps = tokenizer(inps, add_special_tokens=False, return_tensors="pt")
print(inps)

input_ids = inps.input_ids.to(device)
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=1,
        top_p=1.0,
        top_k=1,
        do_sample=False,
        use_cache=True
    )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
```

## test_llama2_embedding.py

测试输入不是iput_ids而是Imput_embed。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_path = "/data/gongoubo/VQA/LLaVA/model_hub/shakechen/Llama-2-7b-chat-hf"
model_path = "/data/gongoubo/Qwen-1.5-Factory/model_hub/qwen/Qwen1___5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16)

# 第一种方式，直接将embedding的权重从模型里面拿出来
embedding_weight = None
for name, module in model.named_parameters():
    print(name, module.shape)
    if "embed_tokens" in name:
        embedding_weight = module

print(config)

device = "cuda:0"
model = model.to(device).eval()

conversation = [{"role": "user", "content": "1+1等于几？"}]

inps = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

print(inps)

inps = tokenizer(inps, add_special_tokens=False, return_tensors="pt")
print(inps)

import torch.nn as nn
class Llama2Embedding(nn.Module):
    def __init__(self, config, embedding_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_embeddings = config.vocab_size
        embedding_dim = config.hidden_size
        self.embed_tokens = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embedding_weight = embedding_weight
        self.init_weight()

    def forward(self, x):
        embedding = self.embed_tokens(x)
        return embedding

    def init_weight(self):
        self.embed_tokens.weight.data = self.embedding_weight

input_ids = inps.input_ids
input_ids = input_ids.to(model.device)
        
llama2Embedding = Llama2Embedding(config, embedding_weight)
llama2Embedding = llama2Embedding.half().to(model.device)
llama2Embedding.eval()


# 输入是input_ids的方式，但是不调用generate方法
####################################
max_length = 50  # 设置生成的最大长度

# 初始化输出
generated_sequence = input_ids

for _ in range(max_length):
    with torch.no_grad():
        # 调用模型
        model_output = model(input_ids=generated_sequence)

        # 获取 logits
        logits = model_output.logits[:, -1, :]  # 只取最后一个时间步的 logits

        # 计算下一个 token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)  # 获取下一个 token ID
        if next_token == tokenizer.eos_token_id:
            break
        # 更新生成的序列
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

# 解码生成的序列
decoded_output = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
print(decoded_output)
####################################


# 输入是embedding方式，但是不调用generate方法，并且加上kv-cache
####################################
# 获取嵌入层
input_embeddings = model.get_input_embeddings()
max_length = 50  # 设置生成的最大长度

# 初始化输出
generated_sequence = input_ids
generated_sequence_embedding = input_embeddings(input_ids).to(model.device)
print(generated_sequence_embedding.shape)
kv_cache = None
past_key_values=kv_cache
for _ in range(max_length):
    with torch.no_grad():
        # 调用模型
        model_output = model(inputs_embeds=generated_sequence_embedding, past_key_values=kv_cache)

        # 获取 logits
        logits = model_output.logits[:, -1, :]  # 只取最后一个时间步的 logits

        # 计算下一个 token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)  # 获取下一个 token ID
        if next_token == tokenizer.eos_token_id:
            break
        next_token_embedding = input_embeddings(next_token)
        print(next_token_embedding.shape)
        # 更新生成的序列
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
        generated_sequence_embedding = torch.cat((generated_sequence_embedding, next_token_embedding), dim = 1)
        kv_cache = model_output.past_key_values

# 解码生成的序列
decoded_output = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
print(decoded_output)
####################################
    
# 输入是embedding方式，并使用自定义的嵌入曾，但是不调用generate方法，并且加上kv-cache
####################################
# 获取嵌入层
input_embeddings = model.get_input_embeddings()
max_length = 50  # 设置生成的最大长度

# 初始化输出
generated_sequence = input_ids
generated_sequence_embedding = llama2Embedding(input_ids).to(model.device)
print(generated_sequence_embedding.shape)
kv_cache = None
past_key_values=kv_cache
for _ in range(max_length):
    with torch.no_grad():
        # 调用模型
        model_output = model(inputs_embeds=generated_sequence_embedding, past_key_values=kv_cache)

        # 获取 logits
        logits = model_output.logits[:, -1, :]  # 只取最后一个时间步的 logits

        # 计算下一个 token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)  # 获取下一个 token ID
        if next_token == tokenizer.eos_token_id:
            break
        next_token_embedding = llama2Embedding(next_token.to(model.device))
        print(next_token_embedding.shape)
        # 更新生成的序列
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
        generated_sequence_embedding = torch.cat((generated_sequence_embedding, next_token_embedding), dim = 1)
        kv_cache = model_output.past_key_values

# 解码生成的序列
decoded_output = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
print(decoded_output)
####################################
```

## test_llava.py

接下来就是将上面的都组合起来：

```python
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

```

说明：

- 将<image>作为特殊token加入的词表里面。
- 数据预处理的时候先将带有<image>token的文本用tokenizer编码成Input_ids，然后再用文本编码器转换为text_embedding。然后将图片用clip编码成image_embedding，再将Image_embedding的维度用projector映射成text_embedding的维度，再将text_embedding用<image>的索引进行分割，将image_embedding插入到分割后的embedding的中间，拼接成带有图片embedding的文本embedding输入给语言模型进行回答即可。

# 补充

- test_dataset.py里面不完整，可以忽略。
- 如果缺少对应的模型权重，搜索一下然后下载到相应的地方即可。
