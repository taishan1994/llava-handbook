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


