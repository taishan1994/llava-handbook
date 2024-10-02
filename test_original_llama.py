import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# j接下来我们先取出llava里面语言编码器的权重并用于进行语言的生成
llama_model_path = "/data/gongoubo/VQA/LLaVA/model_hub/shakechen/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
config = AutoConfig.from_pretrained(llama_model_path)


init_llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.float16, device_map="auto")


conversation = [{"role": "user", "content": "who are you?"}]

inps = llama_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print(inps)

inps = llama_tokenizer(inps, add_special_tokens=False, return_tensors="pt")
print(inps)


output = init_llama_model.generate(
    input_ids=inps.input_ids.to(init_llama_model.device),
    max_new_tokens = 512,
)
print(output)
print(llama_tokenizer.decode(output[0], skip_special_tokens=False))