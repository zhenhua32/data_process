"""
批量请求接口
"""

import argparse

from config.parse_config import Config


default_config_file = "config/config.yaml"

parser = argparse.ArgumentParser(description="批量运行")
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=default_config_file,
    help="配置文件路径",
)
# 需要运行的接口名
parser.add_argument(
    "-n",
    "--name",
    type=str,
    required=True,
    help="需要运行的接口名",
)
# 配置覆盖, 可以指定多个参数, 结构需要是 key=val 的形式
parser.add_argument(
    "-o",
    "--override",
    nargs="+",
    type=str,
    default=[],
    help="配置覆盖, 可以指定多个参数, 结构需要是 key=val 的形式",
)

args = parser.parse_args()
print(args)
