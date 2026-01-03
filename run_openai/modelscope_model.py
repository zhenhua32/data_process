"""
调用 modelscope 的模型
每天上限 2000 次, 单个模型上限 500 次

aigc 的模型有额外限制, 具体可以参考响应头.

响应头	描述	示例值
modelscope-ratelimit-requests-limit	用户当天限额	2000
modelscope-ratelimit-requests-remaining	用户当天剩余额度	500
modelscope-ratelimit-model-requests-limit	模型当天限额	500
modelscope-ratelimit-model-requests-remaining	模型当天剩余额度	20

https://www.modelscope.cn/docs/model-service/API-Inference/intro
"""

import requests
import time
import json
from PIL import Image
from io import BytesIO
import os
import dotenv

dotenv.load_dotenv(".env")


base_url = "https://api-inference.modelscope.cn/"
api_key = os.getenv("MODELSCOPE_API_KEY")  # ModelScope Token

common_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}


def create_image_generations(prompt: str, size: str = "2048x2048", model: str = "Qwen/Qwen-Image-2512"):
    """
    创建生图任务, 获取 task_id

    :param prompt: 说明
    :type prompt: str
    :param size: 说明
    :type size: str
    :param model: 说明
    :type model: str
    :return: 说明
    :rtype: tuple[Any, dict[str, Any]]
    """
    response = requests.post(
        f"{base_url}v1/images/generations",
        headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
        data=json.dumps(
            {
                "model": model,  # ModelScope Model-Id, required
                # "loras": "<lora-repo-id>", # optional lora(s)
                # """
                # LoRA(s) Configuration:
                # - for Single LoRA:
                #   "loras": "<lora-repo-id>"
                # - for Multiple LoRAs:
                #   "loras": {"<lora-repo-id1>": 0.6, "<lora-repo-id2>": 0.4}
                # - Upto 6 LoRAs, all weight-coefficients must sum to 1.0
                # """
                "size": size,
                "prompt": prompt,  # required
            },
            ensure_ascii=False,
        ).encode("utf-8"),
    )

    response.raise_for_status()

    # 获取剩余次数
    header = response.headers
    model_limit_info = {
        "modelscope_ratelimit_model_requests_limit": header.get("Modelscope-Ratelimit-Model-Requests-Limit"),
        "modelscope_ratelimit_model_requests_remaining": header.get("Modelscope-Ratelimit-Model-Requests-Remaining"),
        "modelscope_ratelimit_requests_limit": header.get("Modelscope-Ratelimit-Requests-Limit"),
        "modelscope_ratelimit_requests_remaining": header.get("Modelscope-Ratelimit-Requests-Remaining"),
    }

    task_id = response.json()["task_id"]

    return task_id, model_limit_info


def get_task_result(task_id: str):
    """
    获取任务结果

    :param task_id: 说明
    :type task_id: str
    :return: 说明
    :rtype: tuple[str, dict[str, Any]]
    """
    while True:
        result = requests.get(
            f"{base_url}v1/tasks/{task_id}",
            headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
        )
        result.raise_for_status()
        data = result.json()

        if data["task_status"] == "SUCCEED":
            image = Image.open(BytesIO(requests.get(data["output_images"][0]).content))
            return image, data
        elif data["task_status"] == "FAILED":
            print("Image Generation Failed.")
            return None, data

        time.sleep(5)


if __name__ == "__main__":
    prompt = "少女怀春, 樱花飘零, 阳光洒落, 细腻画风, 高清, 4K"
    task_id, limit_info = create_image_generations(prompt, size="2048x2048")
    print("Task ID:", task_id)
    print("Limit Info:", limit_info)

    image, result_data = get_task_result(task_id)
    if image:
        image.save("result_image.jpg")
        print("Image saved as result_image.jpg")
