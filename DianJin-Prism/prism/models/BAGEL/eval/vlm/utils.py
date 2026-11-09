# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import yaml

from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, 
    Bagel, 
    Qwen2Config, 
    Qwen2ForCausalLM, 
    SiglipVisionConfig, 
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file

from data.transforms import ImageTransform


def load_model_and_tokenizer(args):
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module ="Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    del model_state_dict
    model = model.cuda().eval()

    return model, tokenizer, new_token_ids


def build_transform():
    with open("./data/configs/example.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    max_image_size = data_config['vlm_sft']['image_transform_args']['max_image_size']
    min_image_size = data_config['vlm_sft']['image_transform_args']['min_image_size']
    image_stride = data_config['vlm_sft']['image_transform_args']['image_stride']
    max_pixels = data_config['vlm_sft']['image_transform_args']['max_pixels']

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation
