import os
from PIL import Image
from tqdm import tqdm


def convert_to_png(input_dir: str, output_dir: str):
    """
    将输入目录中的所有 .jfif 文件转换为 .png 文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有 .jfif 文件
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".jfif"):
            # 打开 .jfif 文件
            jfif_path = os.path.join(input_dir, filename)
            with Image.open(jfif_path) as img:
                # 定义输出文件路径
                png_filename = os.path.splitext(filename)[0] + ".png"
                png_path = os.path.join(output_dir, png_filename)
                # 保存为 .png 文件
                img.save(png_path, "PNG")
        else:
            # 直接复制
            jfif_path = os.path.join(input_dir, filename)
            png_path = os.path.join(output_dir, filename)
            with open(jfif_path, "rb") as f1:
                with open(png_path, "wb") as f2:
                    f2.write(f1.read())

    print("转换完成！")


def resize_png_image(input_dir: str, output_dir: str, least_size: int = 1024):
    """
    将输入目录中的所有 .png 文件调整为指定尺寸
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".png"):
            png_path = os.path.join(input_dir, filename)
            with Image.open(png_path) as img:
                # 获取原始尺寸
                width, height = img.size
                # 计算新的尺寸
                if width < height:
                    new_width = least_size
                    new_height = int((least_size / width) * height)
                else:
                    new_height = least_size
                    new_width = int((least_size / height) * width)
                # 调整图片大小
                img = img.resize((new_width, new_height), Image.ANTIALIAS)
                # 定义输出文件路径
                png_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_dir, png_filename)
                # 保存为 .png 文件
                img.save(output_path, "PNG")


if __name__ == "__main__":
    # 定义输入和输出目录
    input_dir = r"E:\lora_traiun\yona\dataset\000sd"
    output_dir = r"E:\lora_traiun\yona\dataset\000sd_png_1024"

    resize_png_image(input_dir, output_dir, 1024)
