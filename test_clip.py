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
# ====================


# =====================
# apt install jupyter-core
# pip install notebook
# jupyter notebook --port 7767 --allow-root --ip=192.168.16.6
# =====================

# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1000x667 at 0x7FE6A3304220>

