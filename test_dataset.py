import os
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 32000


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([torch.tensor(instance[key]) for instance in instances]
                                  for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def preprocess_llama_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        system="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides,and assist the user with a variety of tasks using natural language."
) -> Dict:
    # 对于每一个source里面的数据，先将其合并
    input_ids = []
    labels = []
    sources = sources["conversations"]
    # print(sources)
    for i, source in enumerate(sources):
        value = source["value"]
        if i == 0:
            value = f"<<SYS>>\n{system}\n<</SYS>>\n\n{value}"
        if i % 2 == 0:
            _input = "<s>" + f"[INST] {value} [/INST]"
            _input_ids = tokenizer(_input, add_special_tokens=False)["input_ids"]
            _labels = [IGNORE_INDEX] * len(_input_ids)
            # print(_input_ids)
            # print(_labels)
        else:
            _input = f" {value} " + "</s>"
            _input_ids = tokenizer(_input, add_special_tokens=False)["input_ids"]
            _labels = _input_ids

        input_ids += _input_ids
        labels += _labels

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data,
                 image_folder,
                 tokenizer,
                 image_processor):
        super(LazySupervisedDataset, self).__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = data

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.image_processor = image_processor

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_file = self.list_data_dict[i]['image']
        processor = self.image_processor
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        image = processor(images=image, return_tensors='pt')['pixel_values'][0]
        print(image.shape)
        data_dict = preprocess_llama_2(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        # if isinstance(i, int):
        #     data_dict = dict(input_ids=data_dict["input_ids"][0],
        #                      labels=data_dict["labels"][0])
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image

        return data_dict


def encode_image(images, vision_tower=None, projected_model=None):
    output = vision_tower(images, output_hidden_states=True)
    output = output.hidden_states[-2]
    print("clip倒数第二层输出：", output.shape)
    output = projected_model(output)
    print("project输出：", output.shape)
    return output

device = None

def prepare_inputs_labels_for_multimodal(
        input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, vision_tower=None, projected_model=None
    ):
    image_features = encode_image(images, vision_tower=vision_tower, projected_model=projected_model)

    _labels = labels  # 需要生成的文本标签
    _position_ids = position_ids  # 位置编码
    _attention_mask = attention_mask

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                 zip(input_ids, attention_mask)]

    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        # 统计每一个input_ids里面的图片的个数
        print(cur_input_ids)
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        print(cur_image_idx, num_images)
        # 如果没有图片，则是按照语言模型的embedding拼接
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        # 第二位为image_token在input_ids里面的位置，第三个为input_ids的长度
        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
            cur_input_ids.shape[0]]
        print(image_token_indices)
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
        # cur_input_ids_noim被image_token分为两个部分
        # print(cur_input_ids_noim)
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        # 就是把image_embedding加入到input_embedding里面
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                                 dtype=cur_labels.dtype))
        # 三部分合成一个张量
        cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

    # 截断超出范围的
    # Truncate sequences to max length as image embeddings can make the sequence longer
    # tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    # if tokenizer_model_max_length is not None:
    #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    # 组成新的batch
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    # 再进行padding到batch里面的最大长度
    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                          device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                         device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel

    input_datas = [
        {
            "conversations": [
                {"from": "human", "value": "<image>\n为该图片生成标注。"},
                {"from": "gpt", "value": "【二手9成新】作家德富曼诺夫传略"}
            ],
            "id": "00514175-0237",
            "image": "wukong_test_images/00514175-0237.jpg"},
        {
            "conversations": [
                {"from": "human", "value": "<image>\n为该图片生成标注。"},
                {"from": "gpt", "value": "五心鸡肉菌汤大抄手图片 第283张"}],
            "id": "00520268-0497",
            "image": "wukong_test_images/00520268-0497.jpg"},
        {
            "conversations": [
                {"from": "human", "value": "<image>\n为该图片生成标注。"},
                {"from": "gpt", "value": "2018奥迪专业双杯竞赛落幕,练兵售后为产品规划战略打前站"}],
            "id": "00508672-0518",
            "image": "wukong_test_images/00508672-0518.jpg"},
    ]

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


    projected_model = build_projected()
    print(projected_model)

    model_path = "/data/gongoubo/VQA/LLaVA/model_hub/llava-v1.5-7b/mm_projector.bin"

    projected_model_checkpoint = torch.load(model_path)
    # for name, value in projected_model_checkpoint.items():
    #     print(name)
    projected_model = load_projected_params(projected_model, projected_model_checkpoint)


    path = "/data/gongoubo/VQA/LLaVA/model_hub/shakechen/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(path)

    # print(len(tokenizer))
    # tokenizer.add("<image>")
    special_tokens = {
        'additional_special_tokens': ['<image>']  # 可以添加多个特殊符号
    }
    tokenizer.add_special_tokens(special_tokens)
    # 查看添加后的特殊符号
    # print("New special tokens:", tokenizer.all_special_tokens)
    # new_token_id = tokenizer.convert_tokens_to_ids('<image>')
    # print(new_token_id)
    # print(len(tokenizer))
    # print(tokenizer("[INST] <image> \n[/INST]图片里面讲了什么", add_special_tokens=False))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336",
                                            cache_dir="/data/gongoubo/VQA/LLaVA//model_hub/",
                                                   torch_dtype=torch.float32)
    print(vision_tower)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="/data/gongoubo/VQA/LLaVA//model_hub/")
    image_folder = "/data/gongoubo/VQA/LLaVA/playground/data/"
    dataset = LazySupervisedDataset(input_datas,
                                    image_folder,
                                    tokenizer,
                                    processor)
    # print(dataset[0])
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    loader = DataLoader(dataset, collate_fn=data_collator, batch_size=2, drop_last=True)
    for idx, batch in enumerate(loader):
        for k,v in batch.items():
            print(k, v.shape)

        # output = encode_image(batch["images"], vision_tower, projected_model)
        # print(output.shape)
        prepare_inputs_labels_for_multimodal(batch["input_ids"], None, None, None, batch["labels"],
        batch["images"], image_sizes=(14, 14), vision_tower=vision_tower, projected_model=projected_model)
        break