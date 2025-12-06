"""
生成可以导入的 prompts 文件
"""

import os
import json
import base64
import uuid
import datetime
from tqdm import tqdm


def get_template_prompt(
    title: str, content: str, content_translation: str, preview_image: str, reference_image: str, order: int
) -> dict:
    data = {
        "id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "contentTranslation": content_translation,
        "category": "04221cba-ae4e-4072-a09b-921a676e7349",
        "tags": [],
        "format": "text",
        "previewImage": preview_image,
        "referenceImage": reference_image,
        "createdAt": datetime.datetime.now().isoformat() + "Z",
        "updatedAt": datetime.datetime.now().isoformat() + "Z",
        "versions": [],
        "isFavorite": False,
        "order": order,
    }
    return data


def generate_prompts_file(input_file="prompts.json", output_file="prompt_manager_import.json", image_dir="./"):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        prompt_info = json.load(f)

    items = prompt_info["items"]
    result_list = []

    for index, item in tqdm(enumerate(items), desc="Processing items"):
        prompts = item.get("prompts", [])
        images = item.get("images", [])
        image_b64s = []

        for image in images:
            image_path = os.path.join(image_dir, image)
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    image_b64 = "data:image/png;base64," + base64.b64encode(img_file.read()).decode("utf-8")
                    image_b64s.append(image_b64)

        if len(prompts) == 1:
            content = prompts[0]
            content_translation = ""
        elif len(prompts) >= 2:
            content = prompts[0]
            content_translation = prompts[1]
        else:
            content = ""
            content_translation = ""

        if len(image_b64s) == 1:
            preview_image = image_b64s[0]
            reference_image = ""
        elif len(image_b64s) >= 2:
            preview_image = image_b64s[0]
            reference_image = image_b64s[1]
        else:
            preview_image = ""
            reference_image = ""

        title = f"{index}"
        result_list.append(
            get_template_prompt(
                title=title,
                content=content,
                content_translation=content_translation,
                preview_image=preview_image,
                reference_image=reference_image,
                order=index,
            )
        )

    # 先加载示例文件, 然后保存
    with open("example.json", "r", encoding="utf-8") as f:
        example_data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f:
        example_data["prompts"] = result_list
        json.dump(example_data, f, ensure_ascii=False, indent=4)

    print(f"Prompts file generated: {output_file}")


if __name__ == "__main__":
    generate_prompts_file()
