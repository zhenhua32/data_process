import os
import json
from tqdm import tqdm
import requests


def get_all_image_urls():
    with open("prompts.json", "r", encoding="utf-8") as f:
        prompt_info = json.load(f)
    items = prompt_info["items"]
    image_urls = []

    image_url_base = "https://opennana.com/awesome-prompt-gallery/"
    for item in items:
        images = item["images"]
        for image in images:
            full_image_url = image_url_base + image
            image_urls.append(full_image_url)

    print(f"Total images found: {len(image_urls)}")
    return image_urls


def download_image(image_urls, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)

    for image_url in tqdm(image_urls):
        file_name = image_url.split("/")[-1]
        save_path = os.path.join(save_dir, file_name)
        if os.path.exists(save_path):
            print(f"Image already exists, skipping: {file_name}")
            continue
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
        except requests.RequestException as e:
            print(f"Failed to download {file_name}: {e}")


def main():
    image_urls = get_all_image_urls()
    download_image(image_urls)


if __name__ == "__main__":
    main()
