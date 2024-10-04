"""
批量打标, 来自 https://github.com/MNeMoNiCuZ/joy-caption-batch/tree/main
"""

import spaces
import gradio as gr
from huggingface_hub import InferenceClient
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    logging as transformers_logging,
)
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from huggingface_hub import hf_hub_download
import requests

# Configuration options
LOW_VRAM_MODE = True  # Option to switch to a model that uses less VRAM
PRINT_CAPTIONS = False  # Option to print captions to the console during inference
PRINT_CAPTIONING_STATUS = False  # Option to print captioning file status to the console
OVERWRITE = True  # Option to allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
RUN_LOCAL = True  # Option to run the script locally, 主要修改了一些模型路径

# Specify input and output folder paths
# INPUT_FOLDER = Path(__file__).parent / "input"
INPUT_FOLDER = Path(r"E:\lora_traiun\yangying\dataset\000output_64_tag")
OUTPUT_FOLDER = INPUT_FOLDER

# LLM Settings
VLM_PROMPT = "A descriptive caption for this image:\n"  # Changing this doesn't seem to matter. Help plz?
TEMPERATURE = 0.5  # Controls the randomness of predictions. Lower values make the output more focused and deterministic, while higher values increase randomness.
TOP_K = 10  # Limits the sampling pool to the top K most likely options at each step. A lower value makes the output more deterministic, while a higher value allows more diversity.
MAX_NEW_TOKENS = 300  # The maximum number of tokens to generate. This limits the length of the generated text.

# Clip path
if RUN_LOCAL:
    CLIP_PATH = r"G:\code\ai\ComfyUI_windows_portable\ComfyUI\models\clip\siglip-so400m-patch14-384"
    CHECKPOINT_PATH = Path(r"G:\code\ai\ComfyUI_windows_portable\ComfyUI\models\Joy_caption")
else:
    CLIP_PATH = "google/siglip-so400m-patch14-384"
    CHECKPOINT_PATH = Path("wpkklhc6")

TITLE = "<h1><center>JoyCaption Pre-Alpha (2024-07-30a)</center></h1>"

# Model paths based on VRAM usage
if LOW_VRAM_MODE:
    if RUN_LOCAL:
        MODEL_PATH = r"G:\code\ai\ComfyUI_windows_portable\ComfyUI\models\LLM\Meta-Llama-3.1-8B-bnb-4bit"
    else:
        MODEL_PATH = "unsloth/llama-3-8b-bnb-4bit"
else:
    MODEL_PATH = "unsloth/Meta-Llama-3.1-8B"

# Suppress warnings if PRINT_CAPTIONING_STATUS is False
if not PRINT_CAPTIONING_STATUS:
    transformers_logging.set_verbosity_error()

print("Captioning Batch Images Initializing...")

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# URL for downloading the image adapter
IMAGE_ADAPTER_URL = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/blob/main/wpkklhc6/image_adapter.pt"


# Function to download the image adapter from a Hugging Face Space
def download_image_adapter(force_download=False):
    file_path = CHECKPOINT_PATH / "image_adapter.pt"
    if force_download or not file_path.exists():
        print(f"Downloading {file_path.name} from Hugging Face Space...")
        url = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/resolve/main/wpkklhc6/image_adapter.pt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {file_path.name} successfully.")
        else:
            print(f"Failed to download {file_path.name}. Status code: {response.status_code}")
            exit(1)  # Exit if download fails
    else:
        print(f"{file_path.name} already exists.")


# Download the image adapter before proceeding
download_image_adapter()


# Class definition for ImageAdapter
class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# Process all images in the input folder recursively
print("Captioning Initializing")
image_files = list(INPUT_FOLDER.rglob("*"))

# Filter the list based on the Overwrite flag
if not OVERWRITE:
    image_files = [
        image_path
        for image_path in image_files
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", "webp"]
        and not (image_path.parent / (image_path.stem + ".txt")).exists()
    ]
else:
    image_files = [
        image_path
        for image_path in image_files
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", "webp"]
    ]

print(f"Found {len(image_files)} files to process in {INPUT_FOLDER}")

if not image_files:
    print("No images to process. Exiting...")
    exit(0)  # Exit the script if there are no images to process

# Load CLIP, model, and other resources only if there are images to process
print("Loading CLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
clip_model = AutoModel.from_pretrained(CLIP_PATH)
clip_model = clip_model.vision_model
clip_model.eval()
clip_model.requires_grad_(False)
clip_model.to("cuda")

# Tokenizer
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
    tokenizer, PreTrainedTokenizerFast
), f"Tokenizer is of type {type(tokenizer)}"

# LLM
print("Loading LLM")
text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
text_model.eval()

# Image Adapter
print("Loading image adapter")
try:
    # 是在这里初始化的, 连接图片和文本的模型
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
    image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu"))
    image_adapter.eval()
    image_adapter.to("cuda")
except (torch.nn.modules.module.ModuleAttributeError, _pickle.UnpicklingError):
    print("The image adapter file is corrupted. Re-downloading...")
    # Force re-download
    download_image_adapter(force_download=True)
    # Try loading again
    image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu"))
    image_adapter.eval()
    image_adapter.to("cuda")


@spaces.GPU()
@torch.no_grad()
def process_images_batch(image_paths: list):
    """
    改造成 batch 推理, 单次传入的图片数量是 batch_size
    """
    torch.cuda.empty_cache()

    # Preprocess images
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    images = clip_processor(images=images, return_tensors="pt").pixel_values
    images = images.to("cuda")
    print("images shape", images.shape)  # [4, 3, 384, 384]

    # Tokenize the prompt
    prompt = tokenizer.encode(
        VLM_PROMPT, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False
    )

    # Embed images
    with torch.amp.autocast_mode.autocast("cuda", enabled=True):
        vision_outputs = clip_model(pixel_values=images, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        print("image_features shape", image_features.shape)  # [4, 729, 1152]
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to("cuda")
        print("embedded_images shape", embedded_images.shape)  # [4, 729, 4096]

    # Embed prompt
    prompt_embeds = text_model.model.embed_tokens(prompt.to("cuda"))
    assert prompt_embeds.shape == (
        1,
        prompt.shape[1],
        text_model.config.hidden_size,
    ), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
    embedded_bos = text_model.model.embed_tokens(
        torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64)
    )

    # Construct prompts
    inputs_embeds = torch.cat(
        [
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ],
        dim=1,
    )
    print("inputs_embeds shape", inputs_embeds.shape)  # [4, 737, 4096]

    input_ids = torch.cat(
        [
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(embedded_images.shape[0], -1),
            torch.zeros((embedded_images.shape[0], embedded_images.shape[1]), dtype=torch.long),
            prompt.expand(embedded_images.shape[0], -1),
        ],
        dim=1,
    ).to("cuda")
    print("input_ids shape", input_ids.shape)  # [4, 737]
    attention_mask = torch.ones_like(input_ids)

    # Generate captions
    generate_ids = text_model.generate(
        input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        suppress_tokens=None,
    )
    print("generate_ids shape", generate_ids.shape)  # [4, 937]

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1] :]
    # 如果生成的句子以 eos 结尾, 则去掉 eos
    # generate_ids[generate_ids[:, -1] == tokenizer.eos_token_id] = generate_ids[:, :-1]

    # Prepend/Append strings to the generated captions
    captions = [
        f"{PREPEND_STRING}{tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)}{APPEND_STRING}".strip()
        for ids in generate_ids
    ]

    # Save captions to text files in the same directory as the images
    for image_path, caption in zip(image_paths, captions):
        output_file_path = image_path.parent / (image_path.stem + ".txt")

        if output_file_path.exists() and not OVERWRITE:
            if PRINT_CAPTIONING_STATUS:
                print(f"Skipping {output_file_path} as it already exists.")
            continue

        if PRINT_CAPTIONING_STATUS:
            print(f"Saving caption to {output_file_path}")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(caption)

        if PRINT_CAPTIONS:
            print(f"Caption for {image_path.name}: {caption}")

    return captions


processed = False

# Process images in batches using tqdm for progress bar
batch_size = 16  # You can adjust the batch size based on your VRAM capacity
process_bar = tqdm(total=len(image_files), desc="Processing images")
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i : i + batch_size]
    captions = process_images_batch(batch_files)
    process_bar.update(len(batch_files))
    processed = True

if not processed:
    print("No images processed. Ensure the folder contains supported image formats.")

if __name__ == "__main__":
    print("Processing all images in the input folder")
