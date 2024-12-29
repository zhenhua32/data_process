#!/usr/bin/env python3
"""
更新的版本, 替换 comfyui_workflow/joytag_caption_batch.py
Use JoyCaption to caption images.
"""
import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path

import PIL.Image
import torch
import torch.amp
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def none_or_type(value, desired_type):
    if value == "None":
        return None
    return desired_type(value)


parser = argparse.ArgumentParser()
parser.add_argument("--glob", type=str, help="Glob pattern to find images")
parser.add_argument("--filelist", type=str, help="File containing list of images")
parser.add_argument("--prompt", type=str, help="Prompt to use")
parser.add_argument("--prompt-file", type=str, help="JSON file containing prompts to use")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--top-p", type=lambda x: none_or_type(x, float), default=0.9, help="Top-p sampling")
parser.add_argument("--top-k", type=lambda x: none_or_type(x, int), default=None, help="Top-k sampling")
parser.add_argument(
    "--max-new-tokens", type=int, default=256, help="Maximum length of the generated caption (in tokens)"
)
parser.add_argument("--num-workers", type=int, default=4, help="Number of workers loading images in parallel")
parser.add_argument("--model", type=str, default="fancyfeast/llama-joycaption-alpha-two-hf-llava", help="Model to use")


PIL.Image.MAX_IMAGE_PIXELS = 933120000  # Quiets Pillow from giving warnings on really large images (WARNING: Exposes a risk of DoS from malicious images)


@dataclasses.dataclass
class Prompt:
    prompt: str
    weight: float


@torch.no_grad()
def main():
    # Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Parse arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    # Make sure we have a prompt or a prompt file
    prompts = parse_prompts(args.prompt, args.prompt_file)

    # Find the images
    image_paths = find_images(args.glob, args.filelist)
    if len(image_paths) == 0:
        logging.warning("No images found")
        return
    logging.info(f"Found {len(image_paths)} images")

    # Ignore all images that already have captions 跳过那些已经有 caption 的图片
    image_paths = [path for path in image_paths if not Path(path).with_suffix(".txt").exists()]

    # Load JoyCaption 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), f"Tokenizer is of type {type(tokenizer)}"
    llava_model = LlavaForConditionalGeneration.from_pretrained(args.model, torch_dtype="bfloat16", device_map=0)
    assert isinstance(llava_model, LlavaForConditionalGeneration)

    # 创建数据集
    dataset = ImageDataset(
        prompts, image_paths, tokenizer, llava_model.config.image_token_index, llava_model.config.image_seq_length
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
    )
    end_of_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)

    pbar = tqdm(total=len(image_paths), desc="Captioning images...", dynamic_ncols=True)
    for batch in dataloader:
        vision_dtype = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        vision_device = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
        language_device = llava_model.language_model.get_input_embeddings().weight.device

        # Move to GPU
        pixel_values = batch["pixel_values"].to(vision_device, non_blocking=True)
        input_ids = batch["input_ids"].to(language_device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(language_device, non_blocking=True)

        # Normalize the image
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(vision_dtype)

        # Generate the captions 生成 caption
        generate_ids = llava_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            suppress_tokens=None,
            use_cache=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        # Trim off the prompts
        assert isinstance(generate_ids, torch.Tensor)
        generate_ids = generate_ids.tolist()
        generate_ids = [trim_off_prompt(ids, end_of_header_id, end_of_turn_id) for ids in generate_ids]

        # Decode the captions
        captions = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        captions = [c.strip() for c in captions]

        for path, caption in zip(batch["paths"], captions):
            # 写入到文件中
            write_caption(Path(path), caption)

        pbar.update(len(captions))


def trim_off_prompt(input_ids: list[int], eoh_id: int, eot_id: int) -> list[int]:
    # Trim off the prompt
    while True:
        # 一直寻找 eoh_id, eoh_id 可能有多个
        try:
            i = input_ids.index(eoh_id)
        except ValueError:
            break

        # 然后丢弃 eoh_id 之前的部分
        input_ids = input_ids[i + 1 :]

    # Trim off the end  寻找 eot_id, 丢弃 eot_id 之后的部分
    try:
        i = input_ids.index(eot_id)
    except ValueError:
        return input_ids

    return input_ids[:i]


def write_caption(image_path: Path, caption: str):
    """写入 caption

    Args:
        image_path (Path): _description_
        caption (str): _description_
    """
    caption_path = image_path.with_suffix(".txt")

    try:
        f = os.open(
            caption_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )  # Write-only, create if not exist, fail if exists
    except FileExistsError:
        logging.warning(f"Caption file '{caption_path}' already exists")
        return
    except Exception as e:
        logging.error(f"Failed to open caption file '{caption_path}': {e}")
        return

    try:
        os.write(f, caption.encode("utf-8"))
        os.close(f)
    except Exception as e:
        logging.error(f"Failed to write caption to '{caption_path}': {e}")
        return


class ImageDataset(Dataset):
    def __init__(
        self,
        prompts: list[Prompt],
        paths: list[Path],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        image_token_id: int,
        image_seq_length: int,
    ):
        self.prompts = prompts
        self.paths = paths
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]

        # Pick a prompt 按权重随机选择一个 prompt
        prompt_str = random.choices(self.prompts, weights=[p.weight for p in self.prompts])[0].prompt

        # Preprocess image 自定义图片处理流程
        # NOTE: I don't use the Processor here and instead do it manually.
        # This is because in my testing a simple resize in Pillow yields higher quality results than the Processor,
        # and the Processor had some buggy behavior on some images.
        # And yes, with the so400m model, the model expects the image to be squished into a square, not padded.
        try:
            image = Image.open(path)
            if image.size != (384, 384):
                image = image.resize((384, 384), Image.LANCZOS)
            image = image.convert("RGB")
            pixel_values = TVF.pil_to_tensor(image)
        except Exception as e:
            logging.error(f"Failed to load image '{path}': {e}")
            pixel_values = None  # Will be filtered out later

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        # Format the conversation
        convo_string = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

        # Repeat the image tokens
        input_tokens = []
        for token in convo_tokens:
            if token == self.image_token_id:
                # 图片 token 重复
                input_tokens.extend([self.image_token_id] * self.image_seq_length)
            else:
                input_tokens.append(token)

        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "path": path,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        # Filter out images that failed to load 过滤上一步中没结果的图片
        batch = [item for item in batch if item["pixel_values"] is not None]

        # Pad input_ids and attention_mask
        # Have to use left padding because HF's generate can't handle right padding it seems
        max_length = max(item["input_ids"].shape[0] for item in batch)
        n_pad = [max_length - item["input_ids"].shape[0] for item in batch]
        # stack 之后的 shape 为 (batch_size, max_length)
        input_ids = torch.stack(
            [
                torch.nn.functional.pad(item["input_ids"], (n, 0), value=self.pad_token_id)
                for item, n in zip(batch, n_pad)
            ]
        )
        attention_mask = torch.stack(
            [torch.nn.functional.pad(item["attention_mask"], (n, 0), value=0) for item, n in zip(batch, n_pad)]
        )

        # Stack pixel values
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        # Paths
        paths = [item["path"] for item in batch]

        return {
            "paths": paths,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def parse_prompts(prompt_str: str | None, prompt_file: str | None) -> list[Prompt]:
    """获取 Prompt 列表

    Args:
        prompt_str (str | None): _description_
        prompt_file (str | None): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        list[Prompt]: _description_
    """
    if prompt_str is not None and prompt_file is not None:
        raise ValueError("Cannot specify both --prompt and --prompt-file")

    # 直接返回列表, 仅有一个元素
    if prompt_str is not None:
        return [Prompt(prompt=prompt_str, weight=1.0)]

    if prompt_file is None:
        raise ValueError("Must specify either --prompt or --prompt-file")

    data = json.loads(Path(prompt_file).read_text())

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of prompts")

    prompts = []

    for item in data:
        if isinstance(item, str):
            # 文本还是同样的处理
            prompts.append(Prompt(prompt=item, weight=1.0))
        elif (
            isinstance(item, dict)
            and "prompt" in item
            and "weight" in item
            and isinstance(item["prompt"], str)
            and isinstance(item["weight"], (int, float))
        ):
            # 字典可以带权重
            prompts.append(Prompt(prompt=item["prompt"], weight=item["weight"]))
        else:
            raise ValueError(
                f"Invalid prompt in JSON file. Should be either a string or an object with 'prompt' and 'weight' fields: {item}"
            )

    if len(prompts) == 0:
        raise ValueError("No prompts found in JSON file")

    # 权重和必须为正数
    if sum(p.weight for p in prompts) <= 0.0:
        raise ValueError("Prompt weights must sum to a positive number")

    return prompts


def find_images(glob: str | None, filelist: str | Path | None) -> list[Path]:
    """获取图片路径

    Args:
        glob (str | None): _description_
        filelist (str | Path | None): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[Path]: _description_
    """
    if glob is None and filelist is None:
        raise ValueError("Must specify either --glob or --filelist")

    paths = []

    if glob is not None:
        paths.extend(Path(".").glob(glob))

    # 按行读取图片路径
    if filelist is not None:
        paths.extend(
            (Path(line.strip()) for line in Path(filelist).read_text().strip().splitlines() if line.strip() != "")
        )

    return paths


if __name__ == "__main__":
    main()
