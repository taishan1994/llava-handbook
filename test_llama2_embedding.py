import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_path = "/data/gongoubo/VQA/LLaVA/model_hub/shakechen/Llama-2-7b-chat-hf"
model_path = "/data/gongoubo/Qwen-1.5-Factory/model_hub/qwen/Qwen1___5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16)

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
        for name, module in self.embed_tokens.named_parameters():
            module.data.weight = self.embedding_weight



llama2Embedding = Llama2Embedding(config, embedding_weight)

out_embedding = llama2Embedding(inps.input_ids)
print(out_embedding)
embeddings = model.get_input_embeddings()(inps.input_ids.to(model.device))
print(embeddings)

out_embedding = out_embedding.half().to(device)
model_output = model(
    inputs_embeds = embeddings,
)

logits = model_output.logits
logits = torch.argmax(logits, -1)
print(logits)
print(tokenizer.decode(logits[0]))
with torch.no_grad():
    model_output = model(input_ids=inps.input_ids.to(model.device), attention_mask=inps.attention_mask.to(model.device))
    logits = model_output.logits
    logits = torch.argmax(logits, -1)
    print(logits)
    print(tokenizer.decode(logits[0], skip_special_tokens=True))

output = model.generate(
    input_ids=inps.input_ids.to(model.device)
)
print(output)
print(tokenizer.decode(output[0]))

input_ids = inps.input_ids.to(model.device)
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