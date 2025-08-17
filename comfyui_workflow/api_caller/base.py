"""
保存常用 api 的用法
"""

import os
import json
import requests


api_prompt_url = "http://127.0.0.1:8188/prompt"
api_history_url = "http://127.0.0.1:8188/history"
api_view_url = "http://127.0.0.1:8188/view"


def queue_prompt(prompt: dict) -> dict:
    """
    向 ComfyUI 的 API 发送请求，提交一个新的 prompt 任务。

    :param prompt: dict, 包含 ComfyUI 工作流的 JSON 描述
    :return: dict, 包含响应数据
    """
    data = {"prompt": prompt}
    response = requests.post(api_prompt_url, json=data)
    return response.json()


def get_history(prompt_id: str) -> dict:
    """
    获取指定 prompt_id 的历史记录。

    :param prompt_id: str, 要查询的 prompt ID
    :return: dict, 包含历史记录数据
    """
    history_url = f"{api_history_url}/{prompt_id}"
    response = requests.get(history_url)
    return response.json()


def get_image(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """
    获取指定图片的二进制数据。

    :param filename: str, 图片文件名
    :param subfolder: str, 子文件夹名称
    :param folder_type: str, 文件类型（如 output）
    :return: bytes, 图片的二进制数据
    """
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(api_view_url, params=params)
    return response.content





