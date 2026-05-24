"""
操作标签
"""

import os
import shutil
from PIL import Image
from tqdm import tqdm


def replace_txt_content(input_dir: str, output_dir: str, new_content: str = "gbt"):
    """
    复制数据集目录, 并将所有的标签重置为 "gbt"
    """
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            # txt_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            # 打开文件并写入新内容
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        elif filename.endswith(".png"):
            shutil.copy(os.path.join(input_dir, filename), output_dir)

    print("所有 .txt 文件内容已更新并复制，所有 .png 文件已复制！")


if __name__ == "__main__":
    # 定义输入和输出目录
    input_dir = r"E:\lora_traiun\jingtian\dataset\000output_120"
    output_dir = r"E:\lora_traiun\jingtian\dataset\000output_120_gbt"
    replace_txt_content(input_dir, output_dir)
