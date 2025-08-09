"""
保存常用 api 的用法
"""

import os
import json
import requests


api_prompt_url = "http://127.0.0.1:8188/prompt"


def queue_prompt(prompt: dict) -> dict:
    """
    向 ComfyUI 的 API 发送请求，提交一个新的 prompt 任务。

    :param prompt: dict, 包含 ComfyUI 工作流的 JSON 描述
    :return: dict, 包含响应数据
    """
    data = {"prompt": prompt}
    response = requests.post(api_prompt_url, json=data)

    return response.json()
