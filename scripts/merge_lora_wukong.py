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
